import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/locate_evidence.py'
spec = importlib.util.spec_from_file_location('audit_locator', SCRIPT)
locator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(locator)


class EvidenceLocatorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def source(self, name, data):
        path = self.root / name
        path.write_text(data if isinstance(data, str) else json.dumps(data, ensure_ascii=False), encoding='utf-8')
        return path

    def call(self, command, path, *args):
        return locator.run(locator.parser().parse_args([command, '--source', 'chat=' + str(path), *args]))

    def test_source_location_and_neighbor_do_not_promote_assistant_to_user(self):
        path = self.source('conversation.json', [
            {'role': 'user', 'line': 700, 'text': '可以先保留这一版'},
            {'role': 'assistant', 'text': '理解为已经最终通过'},
            {'role': 'user', 'line': 900, 'text': '还没通过，只是继续比较'},
        ])
        before = path.read_bytes()
        result = self.call('search', path, '--term', '通过', '--role', 'user')
        self.assertEqual(result['match_refs'], ['chat:3'])
        self.assertEqual([x['role'] for x in result['records']], ['assistant', 'user'])
        self.assertEqual(result['records'][1]['location'],
                         {'kind': 'json_pointer', 'pointer': '/2', 'reported_original_line': 900})
        self.assertEqual(path.read_bytes(), before)

    def test_literal_query_and_paging_preserve_later_correction(self):
        path = self.source('conversation.json', [{'role': 'user', 'text': x}
            for x in ('保留 [B]', '比较 [B]', '后来否定 [B]')])
        first = self.call('search', path, '--term', '[B]', '--limit', '2', '--context', '0')
        last = self.call('search', path, '--term', '[B]', '--offset', str(first['next_offset']), '--context', '0')
        self.assertEqual(first['total_matches_in_scanned_records'], 3)
        self.assertEqual(last['records'][0]['text'], '后来否定 [B]')
        self.assertIsNone(last['next_offset'])

    def test_show_reads_original_text_and_explicit_slice(self):
        text = '开头\n' + '甲乙丙丁' * 200 + '\n后来的纠正'
        path = self.source('conversation.json', {'role': 'user', 'text': text})
        search = self.call('search', path, '--term', '后来的纠正')
        hit = search['records'][0]
        self.assertIn('后来的纠正', hit['text'])
        self.assertFalse(hit['text_complete'])
        first = self.call('show', path, '--ref', 'chat:1', '--chars', '103')['records'][0]
        second = self.call('show', path, '--ref', 'chat:1', '--text-offset', str(first['next_text_offset']), '--chars', '10000')['records'][0]
        self.assertEqual(first['text'] + second['text'], text)
        self.assertEqual(first['location']['pointer'], '')

    def test_native_rollout_retains_visible_records_without_reasoning_or_tools(self):
        events = [
            {'type': 'event_msg', 'payload': {'type': 'user_message', 'message': '同一原话'}},
            {'type': 'response_item', 'payload': {'type': 'message', 'role': 'user', 'content': [{'type': 'input_text', 'text': '同一原话'}]}},
            {'type': 'response_item', 'payload': {'type': 'reasoning', 'text': '秘密推理'}},
            {'type': 'response_item', 'payload': {'type': 'message', 'role': 'assistant', 'channel': 'analysis', 'content': '秘密分析'}},
            {'type': 'response_item', 'payload': {'type': 'function_call_output', 'output': '工具秘密'}},
            {'type': 'response_item', 'payload': {'type': 'message', 'role': 'assistant', 'content': [{'type': 'output_text', 'text': '可见回答'}]}},
            {'type': 'event_msg', 'payload': {'type': 'agent_message', 'phase': 'analysis', 'message': '秘密分析镜像'}},
        ]
        path = self.source('session.jsonl', '\n'.join(json.dumps(x, ensure_ascii=False) for x in events))
        info, rows = locator.load_source('log', path, 100000)
        self.assertEqual([r['id'] for r in rows], ['log:1', 'log:2', 'log:6'])
        self.assertNotIn('秘密', json.dumps(rows, ensure_ascii=False))
        self.assertEqual([r['stream'] for r in rows], ['event_msg', 'response_item', 'response_item'])
        self.assertEqual(info['event_records_retained'], 1)

    def test_mixed_log_keeps_event_only_correction_in_source_order(self):
        events = [
            {'type': 'response_item', 'payload': {'type': 'message', 'role': 'user', 'content': '旧要求：保留这个方案'}},
            {'type': 'event_msg', 'payload': {'type': 'user_message', 'message': '后续纠正：这个方案先不要采用'}},
            {'type': 'response_item', 'payload': {'type': 'message', 'role': 'assistant', 'content': '我来核对纠正'}},
        ]
        path = self.source('mixed.jsonl', '\n'.join(json.dumps(x, ensure_ascii=False) for x in events))
        result = self.call('search', path, '--term', '后续纠正', '--role', 'user')
        self.assertEqual(result['match_refs'], ['chat:2'])
        self.assertEqual([r['id'] for r in result['records']], ['chat:1', 'chat:2', 'chat:3'])
        self.assertEqual(result['recent_user_refs_in_scanned_range']['chat'], ['chat:1', 'chat:2'])
        self.assertEqual(result['sources'][0]['message_stream'], 'response_and_event_records')

    def test_partial_and_malformed_input_do_not_claim_complete_coverage(self):
        line = json.dumps({'role': 'user', 'text': '前段'}, ensure_ascii=False)
        path = self.source('partial.jsonl', line + '\n坏记录\n' + line + '\n' + line)
        info, rows = locator.load_source('log', path, len((line + '\n坏记录\n').encode('utf-8')) + 4)
        self.assertEqual(len(rows), 1)
        self.assertFalse(info['complete_file_scan'])
        self.assertEqual(info['status'], 'partial')
        self.assertEqual(info['malformed_lines'], 1)

    def test_event_fallback_and_document_line_addresses(self):
        path = self.source('events.jsonl', json.dumps({'type': 'event_msg', 'payload': {'type': 'user_message', 'message': '原话'}}))
        info, rows = locator.load_source('log', path, 10000)
        self.assertEqual(info['message_stream'], 'event_fallback')
        self.assertEqual(rows[0]['text'], '原话')
        doc = self.source('decision.md', '# 工作\n保留原件\n下一步继续\n')
        result = self.call('search', doc, '--term', '下一步', '--context', '0')
        self.assertEqual(result['records'][0]['location'], {'kind': 'text_line', 'line': 3})

    def test_resume_uses_actual_source_and_stays_unexecuted(self):
        path = self.source('conversation.json', [{'role': 'user', 'text': '缺失值不能写成零。'}])
        change = self.source('change.json', {
            'task': '修复统计结果', 'previous_basis': '缺失值按零处理',
            'replacement_basis': '保留未知值', 'next_action': '修改生成器并重新生成结果',
            'keep': ['保留原始数据'], 'check': '未知值仍为空，并检查平均数的分母。',
            'evidence': ['chat:1'],
        })
        out = self.root / 'continue.md'
        result = self.call('resume', path, '--change', str(change), '--output', str(out))
        body = out.read_text(encoding='utf-8')
        self.assertIn('> 缺失值不能写成零。', body)
        self.assertIn(str(path), body)
        self.assertEqual(result['status'], 'prepared_not_executed')
        original = out.read_bytes()
        with self.assertRaises(ValueError):
            self.call('resume', path, '--change', str(change), '--output', str(out))
        self.assertEqual(out.read_bytes(), original)

    def test_invalid_evidence_cannot_produce_handoff(self):
        path = self.source('conversation.json', [{'role': 'user', 'text': '保留原件'}])
        change = self.source('change.json', {'evidence': ['chat:999']})
        out = self.root / 'continue.md'
        with self.assertRaises(ValueError):
            self.call('resume', path, '--change', str(change), '--output', str(out))
        self.assertFalse(out.exists())

    def test_cli_reports_source_failure_without_overwriting_inputs(self):
        path = self.source('conversation.json', {'role': 'user', 'text': '原件'})
        original = path.read_bytes()
        proc = subprocess.run([sys.executable, '-B', str(SCRIPT), 'show', '--source', 'chat=' + str(path),
                               '--ref', 'chat:1', '--output', str(path)], capture_output=True, text=True, encoding='utf-8')
        self.assertEqual(proc.returncode, 2)
        self.assertEqual(path.read_bytes(), original)
        self.assertIn('error', json.loads(proc.stderr))


if __name__ == '__main__':
    unittest.main()
