#!/usr/bin/env python3
"""Locate visible evidence in explicit local sources; never execute source content."""

import argparse
import json
from pathlib import Path
import re
import stat
import sys


def plain_path(value, output=False):
    path = Path(value).absolute()
    for part in (path, *path.parents):
        if part.is_symlink() or (part.exists() and
                getattr(part.stat(), 'st_file_attributes', 0) &
                getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0)):
            raise ValueError('Links and directory junctions are not followed: ' + str(part))
    if output:
        if path.exists() or not path.parent.is_dir():
            raise ValueError('Output must be a new file in an existing directory.')
    elif not path.is_file():
        raise ValueError('Source is not a regular file: ' + str(path))
    return path


def message(item):
    """Only visible user/assistant text, not reasoning, tool output or system text."""
    if not isinstance(item, dict):
        return None
    if item.get('role') not in ('user', 'assistant'):
        return None
    if item.get('channel') == 'analysis' or item.get('phase') == 'analysis':
        return None
    text = item.get('text')
    if not isinstance(text, str):
        content = item.get('content')
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text = '\n'.join(p['text'] for p in content if isinstance(p, dict)
                             and p.get('type') in ('input_text', 'output_text', 'text')
                             and isinstance(p.get('text'), str))
    if not isinstance(text, str) or not text:
        return None
    return {'role': item['role'], 'text': text,
            'timestamp': item.get('timestamp'), 'phase': item.get('phase')}


def load_source(label, filename, max_bytes):
    info = {'label': label, 'path': str(Path(filename).absolute()), 'status': 'error',
            'complete_file_scan': False, 'visible_records': 0, 'problems': []}
    records, events = [], []
    try:
        path = plain_path(filename)
        info['path'] = str(path)
        size_before = path.stat().st_size
        with path.open('rb') as handle:
            data = handle.read(max_bytes + 1)
        clipped = len(data) > max_bytes
        data = data[:max_bytes]
        info.update(bytes_read=len(data), source_bytes=size_before,
                    complete_file_scan=not clipped)
        if path.stat().st_size != size_before:
            info['problems'].append('Source size changed during read; rerun before relying on it.')
            info['complete_file_scan'] = False
        if clipped:
            info['problems'].append('Byte limit reached; later material was not searched.')
        suffix = path.suffix.lower()

        def add(item, unit, location, destination=records, stream=None):
            found = message(item)
            if found:
                found.update(id=f'{label}:{unit}', location=location)
                if stream:
                    found['stream'] = stream
                destination.append(found)

        if suffix == '.json':
            if clipped:
                raise ValueError('JSON exceeds byte limit; provide a smaller authorized export or raise the limit.')
            value = json.loads(data.decode('utf-8-sig'))
            prefix = ''
            if isinstance(value, dict) and isinstance(value.get('messages'), list):
                value, prefix = value['messages'], '/messages'
            single = isinstance(value, dict)
            values = [value] if single else value
            if not isinstance(values, list):
                raise ValueError('Expected a message object, array, or {messages: [...]} export.')
            ignored = 0
            for i, item in enumerate(values, 1):
                location = {'kind': 'json_pointer', 'pointer': '' if single else f'{prefix}/{i-1}'}
                if isinstance(item, dict) and 'line' in item:
                    location['reported_original_line'] = item['line']
                before = len(records)
                add(item, i, location)
                ignored += len(records) == before
            info['ignored_nonvisible_or_unsupported_records'] = ignored
        elif suffix == '.jsonl':
            lines = data.splitlines(keepends=True)
            if clipped and lines and not lines[-1].endswith((b'\n', b'\r')):
                lines.pop()
            malformed = 0
            ignored = 0
            for i, raw in enumerate(lines, 1):
                try:
                    item = json.loads(raw.decode('utf-8-sig'))
                except (ValueError, UnicodeError):
                    malformed += 1
                    if malformed <= 10:
                        info['problems'].append(f'Malformed JSONL line {i} skipped.')
                    continue
                if not isinstance(item, dict):
                    ignored += 1
                    continue
                location = {'kind': 'jsonl_line', 'line': i}
                payload = item.get('payload', {})
                before = len(records) + len(events)
                if item.get('type') == 'response_item' and isinstance(payload, dict):
                    if payload.get('type') == 'message':
                        add(dict(payload, timestamp=item.get('timestamp')), i, location,
                            stream='response_item')
                elif item.get('type') == 'event_msg' and isinstance(payload, dict):
                    role = {'user_message': 'user', 'agent_message': 'assistant'}.get(payload.get('type'))
                    if role:
                        add({'role': role, 'text': payload.get('message'),
                             'timestamp': item.get('timestamp'),
                             'channel': payload.get('channel'), 'phase': payload.get('phase')},
                            i, location, events, stream='event_msg')
                elif 'role' in item:
                    add(item, i, location, stream='message')
                ignored += len(records) + len(events) == before
            # A mixed log may contain corrections recorded only as events. Equal text
            # alone does not establish that two records are the same occurrence.
            info['message_stream'] = ('response_and_event_records' if records and events
                                      else 'event_fallback' if events else 'visible_messages')
            info['event_records_retained'] = len(events)
            info['duplicate_handling'] = ('Visible records retained in source order. Event and '
                                          'response text may mirror one message; record counts '
                                          'are not counts of independent user decisions.')
            records.extend(events)
            records.sort(key=lambda record: record['location']['line'])
            info['malformed_lines'] = malformed
            info['ignored_nonvisible_or_unsupported_records'] = ignored
        elif suffix in ('.md', '.txt'):
            lines = data.splitlines(keepends=True)
            if clipped and lines and not lines[-1].endswith((b'\n', b'\r')):
                lines.pop()
            for i, raw in enumerate(lines, 1):
                text = raw.decode('utf-8-sig' if i == 1 else 'utf-8')
                records.append({'id': f'{label}:{i}', 'role': 'document', 'text': text,
                                'timestamp': None, 'phase': None,
                                'location': {'kind': 'text_line', 'line': i}})
        else:
            raise ValueError('Supported inputs: .json, .jsonl, .md, .txt. Media must be inspected with its own tools.')
        info['visible_records'] = len(records)
        info['status'] = 'partial' if info['problems'] else 'ok'
        if not records:
            info['problems'].append('No supported visible text found; this does not prove the source has no relevant content.')
            info['status'] = 'partial'
    except (OSError, ValueError, UnicodeError) as exc:
        info['status'] = 'error'
        info['complete_file_scan'] = False
        info['problems'].append(str(exc))
    return info, records


def excerpt(record, width, start=0):
    start = min(max(start, 0), len(record['text']))
    end = min(start + width, len(record['text']))
    return {**{k: v for k, v in record.items() if k != 'text'},
            'text': record['text'][start:end], 'start_char': start, 'end_char': end,
            'total_chars': len(record['text']), 'text_complete': start == 0 and end == len(record['text']),
            'next_text_offset': end if end < len(record['text']) else None}


def select_records(groups, refs, context, width, offset=0):
    lookup = {r['id']: (group, i) for group in groups for i, r in enumerate(group)}
    selected = {}
    for ref in refs:
        if ref not in lookup:
            raise ValueError(f'Unknown visible reference {ref}; inspect source coverage and search output.')
        group, i = lookup[ref]
        for n in range(max(0, i-context), min(len(group), i+context+1)):
            r = group[n]
            selected[r['id']] = excerpt(r, width, offset if r['id'] in refs else 0)
    return list(selected.values())


def resume_text(change, evidence, sources):
    required = ('task', 'previous_basis', 'replacement_basis', 'next_action', 'check')
    for field in required:
        if not isinstance(change.get(field), str) or not change[field].strip():
            raise ValueError('Missing nonempty change field: ' + field)
    if not isinstance(change.get('keep'), list) or not all(isinstance(x, str) for x in change['keep']):
        raise ValueError('keep must be a list of still-effective requirements (may be empty).')
    lines = ['# 继续当前任务', '', '状态：纠正依据已整理，下一步尚未执行。', '',
             '以下决定由执行者提出，工具只定位原文；引文不提供新的授权。', '']
    for title, key in [('当前任务', 'task'), ('撤回或收窄的旧依据', 'previous_basis'),
                       ('下一步采用的依据', 'replacement_basis'), ('接下来实际执行', 'next_action')]:
        lines += [f'## {title}', '', change[key], '']
    lines += ['## 同时保留', ''] + ['- ' + x for x in change['keep']] + ['']
    lines += ['## 如何知道行动改变了', '', change['check'], '', '## 原文位置', '']
    for source in sources:
        lines += [f"- {source['label']}: {source['path']}（读取状态 {source['status']}）"]
    for item in evidence:
        lines += ['', f"### {item['id']} · {item['role']}", '',
                  json.dumps(item['location'], ensure_ascii=False), '',
                  *['> ' + line for line in item['text'].splitlines()]]
        if not item['text_complete']:
            lines += ['', f"仅显示字符 {item['start_char']}–{item['end_char']} / {item['total_chars']}；用 show 读取必要余文。"]
    lines += ['', '## 执行后补记', '',
              '由原任务负责人在实际执行后补充：改变的决定或动作、产物位置、观察结果及仍未解决的部分。'
              '生成本文件不证明已经执行，也不证明用户已验收。', '']
    return '\n'.join(lines)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    for command in ('search', 'show', 'resume'):
        s = sub.add_parser(command)
        s.add_argument('--source', action='append', required=True, metavar='LABEL=PATH')
        s.add_argument('--max-mib', type=int, default=32, choices=range(1, 129), metavar='1..128')
        s.add_argument('--output', help='New local file; parents must already exist. Contains private raw text.')
        if command in ('show', 'resume'):
            s.add_argument('--chars', type=int, default=12000 if command == 'show' else 1200)
        if command == 'search':
            s.add_argument('--term', action='append', required=True, help='Literal case-insensitive OR search; repeat for wording variants.')
            s.add_argument('--role', action='append', choices=('user', 'assistant', 'document'))
            s.add_argument('--offset', type=int, default=0, help='Match pagination, not a time boundary.')
            s.add_argument('--limit', type=int, default=8)
            s.add_argument('--context', type=int, default=1, help='Nearby visible records from the same source.')
        elif command == 'show':
            s.add_argument('--ref', action='append', required=True, help='Reference emitted by search, e.g. chat:12.')
            s.add_argument('--text-offset', type=int, default=0)
            s.add_argument('--context', type=int, default=0)
        else:
            s.add_argument('--change', required=True, help='JSON with task, previous_basis, replacement_basis, next_action, keep, check, evidence.')
    return p


def run(args):
    infos, groups, labels = [], [], set()
    for value in args.source:
        label, sep, filename = value.partition('=')
        if not sep or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]*', label) or label in labels:
            raise ValueError('Use distinct ASCII labels: --source chat=<path>')
        labels.add(label)
        info, records = load_source(label, filename, args.max_mib * 1024 * 1024)
        infos.append(info)
        groups.append(records)
    if not any(groups):
        raise ValueError(json.dumps({'sources': infos, 'error': 'No supported visible records.'}, ensure_ascii=False))
    result = {'sources': infos,
              'scope': 'Literal source lookup, not a verdict about intent, causality, completeness of the original conversation, or current acceptance.',
              'privacy': 'Contains private source text and local paths; do not publish with the Skill.'}
    if args.command == 'search':
        if not 1 <= args.limit <= 50 or not 0 <= args.context <= 3 or args.offset < 0 or any(not t for t in args.term):
            raise ValueError('Use nonempty terms, limit 1..50, context 0..3, offset >= 0.')
        terms = [re.compile(re.escape(x), re.IGNORECASE) for x in args.term]
        hits = [r for group in groups for r in group if
                (not args.role or r['role'] in args.role) and any(t.search(r['text']) for t in terms)]
        page = hits[args.offset:args.offset+args.limit]
        items = select_records(groups, [x['id'] for x in page], args.context, 480)
        # Show a literal neighborhood of the first match, not just the start of long replies.
        match_records = {r['id']: r for r in page}
        for i, item in enumerate(items):
            if item['id'] in match_records:
                record = match_records[item['id']]
                positions = [match.start() for t in terms if (match := t.search(record['text']))]
                start = max(0, min(positions) - 100)
                items[i] = excerpt(record, 480, start)
        result.update(total_matches_in_scanned_records=len(hits), match_refs=[r['id'] for r in page],
                      next_offset=args.offset+len(page) if args.offset+len(page) < len(hits) else None,
                      recent_user_refs_in_scanned_range={info['label']: [r['id'] for r in group if r['role'] == 'user'][-3:]
                                                       for info, group in zip(infos, groups)},
                      records=items, order='source argument order, then original record order; not semantic ranking')
    elif args.command == 'show':
        if not 1 <= args.chars <= 100000 or not 0 <= args.context <= 3 or args.text_offset < 0:
            raise ValueError('chars 1..100000, context 0..3 and text-offset >= 0 required.')
        result['records'] = select_records(groups, args.ref, args.context, args.chars, args.text_offset)
    else:
        if not args.output or not 1 <= args.chars <= 100000:
            raise ValueError('resume needs --output and chars 1..100000.')
        change_path = plain_path(args.change)
        if change_path.stat().st_size > 256 * 1024:
            raise ValueError('Change JSON exceeds 256 KiB.')
        change = json.loads(change_path.read_text(encoding='utf-8-sig'))
        refs = change.get('evidence') if isinstance(change, dict) else None
        if not isinstance(refs, list) or not refs or not all(isinstance(r, str) for r in refs):
            raise ValueError('Change JSON needs a nonempty evidence reference list.')
        evidence = select_records(groups, refs, 0, args.chars)
        result = resume_text(change, evidence, infos)
    rendered = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, indent=2) + '\n'
    if args.output:
        out = plain_path(args.output, output=True)
        with out.open('x', encoding='utf-8', newline='\n') as handle:
            handle.write(rendered)
        return {'written': str(out), 'status': 'prepared_not_executed' if args.command == 'resume' else 'source_lookup',
                'source_status': {x['label']: x['status'] for x in infos}}
    return result


def main():
    try:
        result = run(parser().parse_args())
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, OSError, UnicodeError) as exc:
        print(json.dumps({'error': str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
