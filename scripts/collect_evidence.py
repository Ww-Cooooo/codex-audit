#!/usr/bin/env python3
"""Bounded, explicit-input Codex metadata inventory. No network or diagnosis."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import sys
import tomllib

SMALL_LIMIT = 256 * 1024
LINE_LIMIT = 1024 * 1024
LINE_COUNT_LIMIT = 50_000
ROOT_ENTRY_LIMIT = 200
EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra"}
USAGE_KEYS = ("input_tokens", "cached_input_tokens", "output_tokens",
              "reasoning_output_tokens", "total_tokens")
RECORD_TYPES = {"session_meta", "turn_context", "response_item", "event_msg", "compacted"}


def linked(path):
    """Reject symlinks and Windows reparse points, including linked ancestors."""
    for part in (path, *path.parents):
        try:
            if part.is_symlink() or getattr(part.lstat(), "st_file_attributes", 0) & 0x400:
                return True
        except FileNotFoundError:
            continue
    return False


def error_status(exc):
    if isinstance(exc, FileNotFoundError):
        return "missing"
    if isinstance(exc, PermissionError):
        return "unreadable"
    return "io_error"


def basic_source(path, source_id, kind):
    result = {"id": source_id, "kind": kind, "status": "ok", "limits": []}
    try:
        if linked(path):
            result["status"] = "linked_path_skipped"
        elif not path.is_file():
            result["status"] = "not_a_file" if path.exists() else "missing"
        else:
            result["size_bytes"] = path.stat().st_size
    except OSError as exc:
        result["status"] = error_status(exc)
    return result


def small_text(path, result):
    if result["status"] != "ok":
        return None
    if result["size_bytes"] > SMALL_LIMIT:
        result["status"] = "too_large"
        result["limits"].append("small_file_limit_256_KiB")
        return None
    try:
        with path.open("rb") as stream:
            raw = stream.read(SMALL_LIMIT + 1)
        if len(raw) > SMALL_LIMIT:
            result["status"] = "too_large"
            result["limits"].append("file_grew_past_limit")
            return None
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        result["status"] = "invalid_utf8"
        return None
    except OSError as exc:
        result["status"] = error_status(exc)
        return None
    result["bytes_read"] = len(raw)
    return text, raw


def model_value(value):
    if not isinstance(value, str):
        return None
    # Never forward arbitrary provider identifiers, paths, URLs or free text.
    if re.fullmatch(r"gpt-[a-zA-Z0-9.-]{1,64}", value):
        return value
    return "custom_or_unrecognized"


def enum_value(value, choices):
    return value if isinstance(value, str) and value in choices else "unrecognized"


def config_facts(data):
    facts = {}
    if "model" in data:
        facts["model"] = model_value(data["model"])
    for key, choices in (("model_reasoning_effort", EFFORTS),
                         ("service_tier", {"auto", "default", "flex", "priority"})):
        if key in data:
            facts[key] = enum_value(data[key], choices)
    for key in ("project_doc_max_bytes",):
        if type(data.get(key)) is int and 0 <= data[key] <= 10**15:
            facts[key] = data[key]
    memories = data.get("memories")
    if isinstance(memories, dict):
        facts["memories"] = {key: memories[key] for key in ("use_memories", "generate_memories")
                             if type(memories.get(key)) is bool}
    facts["profile_selection_present"] = "profile" in data
    facts["profiles_table_present"] = isinstance(data.get("profiles"), dict)
    facts["provider_configuration_present"] = "model_provider" in data or "model_providers" in data
    facts["instruction_overrides_present"] = any(
        key in data for key in ("instructions", "developer_instructions", "model_instructions_file"))
    return facts


def collect_small(path, source_id, kind):
    result = basic_source(path, source_id, kind)
    value = small_text(path, result)
    if value is None:
        return result
    text, raw = value
    if kind == "instruction":
        result.update(line_count=len(text.splitlines()), sha256=sha256(raw).hexdigest())
    else:
        try:
            data = tomllib.loads(text)
        except (tomllib.TOMLDecodeError, RecursionError):
            result["status"] = "invalid_toml"
            return result
        result["stored_top_level_fields"] = config_facts(data)
        result["limits"].append("stored_fields_only_not_effective_session_configuration")
        if data.get("profiles") or data.get("profile"):
            result["limits"].append("profile_overrides_not_resolved")
    return result


def skill_metadata(text):
    lines = text.splitlines()
    result = {"frontmatter_present": False, "name": None, "description_field_present": False}
    if not lines or lines[0].strip() != "---":
        return result
    end = next((i for i in range(1, len(lines)) if lines[i].strip() == "---"), None)
    if end is None:
        return result
    result["frontmatter_present"] = True
    names = []
    for line in lines[1:end]:
        if line.startswith("name:"):
            name = line[5:].strip()
            if len(name) >= 2 and name[0] == name[-1] and name[0] in "\"'":
                name = name[1:-1]
            names.append(name if re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", name) else None)
        if line.startswith("description:"):
            result["description_field_present"] = True
    if len(names) == 1:
        result["name"] = names[0]
    return result


def collect_skills(path, source_id):
    result = {"id": source_id, "kind": "skill-root", "status": "ok", "limits": [], "skills": []}
    try:
        if linked(path):
            result["status"] = "linked_path_skipped"
            return result
        if not path.is_dir():
            result["status"] = "not_a_directory" if path.exists() else "missing"
            return result
        entries = []
        with os.scandir(path) as listing:
            for entry in listing:
                if len(entries) == ROOT_ENTRY_LIMIT:
                    result["status"] = "partial"
                    result["limits"].append("directory_entry_limit_200")
                    break
                entries.append(Path(entry.path))
        result["entries_examined"] = len(entries)
        skipped_links = 0
        for entry in sorted(entries, key=lambda item: item.name.casefold()):
            if linked(entry):
                skipped_links += 1
                continue
            if not entry.is_dir():
                continue
            skill_file = entry / "SKILL.md"
            if not skill_file.exists() and not skill_file.is_symlink():
                continue
            child_id = f"{source_id}/skill:{len(result['skills']) + 1}"
            child = basic_source(skill_file, child_id, "skill")
            child["relative_path"] = f"{entry.name}/SKILL.md"
            value = small_text(skill_file, child)
            if value is not None:
                child.update(skill_metadata(value[0]))
                if child["name"] is None or not child["description_field_present"]:
                    child["status"] = "metadata_not_recognized"
            result["skills"].append(child)
        if skipped_links:
            result["linked_entries_skipped"] = skipped_links
            result["limits"].append("linked_entries_not_followed")
        if skipped_links or any(item["status"] != "ok" for item in result["skills"]):
            result["status"] = "partial"
    except OSError as exc:
        result["status"] = error_status(exc)
    result["limits"].append("direct_children_only_presence_does_not_prove_loading")
    return result


def usage_fields(value):
    if not isinstance(value, dict):
        return {}
    return {key: value[key] for key in USAGE_KEYS
            if type(value.get(key)) is int and 0 <= value[key] <= 10**15}


def collect_rollout(path, source_id, budget):
    result = basic_source(path, source_id, "rollout")
    if result["status"] != "ok":
        return result
    counts = Counter()
    observed = 0
    lines = 0
    malformed = 0
    unsupported = 0
    context_events = 0
    complete = True
    try:
        with path.open("rb") as stream:
            while True:
                if observed >= budget or lines >= LINE_COUNT_LIMIT:
                    if stream.read(1):
                        complete = False
                        result["limits"].append("prefix_byte_or_line_budget_reached")
                    break
                raw = stream.readline(min(LINE_LIMIT + 1, budget - observed + 1))
                if not raw:
                    break
                if len(raw) > LINE_LIMIT or observed + len(raw) > budget:
                    complete = False
                    result["limits"].append("line_or_prefix_byte_limit_reached")
                    break
                observed += len(raw)
                lines += 1
                try:
                    record = json.loads(raw)
                except (ValueError, UnicodeDecodeError, RecursionError):
                    malformed += 1
                    continue
                if not isinstance(record, dict):
                    unsupported += 1
                    continue
                kind = record.get("type")
                if not isinstance(kind, str) or kind not in RECORD_TYPES:
                    unsupported += 1
                    continue
                counts[kind] += 1
                payload = record.get("payload")
                if not isinstance(payload, dict):
                    unsupported += 1
                    continue
                if kind == "turn_context":
                    model = model_value(payload.get("model"))
                    effort = payload.get("reasoning_effort", payload.get("effort"))
                    fields = {"line": lines}
                    if model is not None:
                        fields["model"] = model
                    if effort is not None:
                        fields["reasoning_effort"] = enum_value(effort, EFFORTS)
                    if len(fields) > 1:
                        result["last_observed_turn_fields"] = fields
                if kind == "event_msg" and payload.get("type") == "context_compacted":
                    context_events += 1
                if kind == "event_msg" and payload.get("type") == "token_count":
                    info = payload.get("info")
                    if not isinstance(info, dict):
                        unsupported += 1
                        continue
                    token_event = {"line": lines}
                    for key in ("last_token_usage", "total_token_usage"):
                        values = usage_fields(info.get(key))
                        if values:
                            token_event[key] = values
                    window = info.get("model_context_window")
                    if type(window) is int and 0 < window <= 10**15:
                        token_event["model_context_window"] = window
                    if len(token_event) > 1:
                        result["last_observed_token_event"] = token_event
                    else:
                        unsupported += 1
    except OSError as exc:
        result["status"] = error_status(exc)
        complete = False
    result.update(bytes_observed=observed, lines_observed=lines,
                  complete_file_read=complete, record_types=dict(sorted(counts.items())),
                  malformed_lines=malformed, unsupported_records=unsupported,
                  context_compacted_event_records=context_events)
    if not complete or malformed or unsupported:
        if result["status"] == "ok":
            result["status"] = "partial"
    if malformed:
        result["limits"].append("malformed_lines_skipped")
    if unsupported:
        result["limits"].append("unsupported_structures_not_interpreted")
    result["limits"].append("observed_metadata_only_not_full_conversation_or_current_state")
    return result


def canonical(path):
    return os.path.normcase(str(path.resolve()))


def build_report(args):
    sources = []
    seen = {}
    names = defaultdict(list)
    for kind in ("config", "instruction", "skill-root", "rollout"):
        for index, value in enumerate(getattr(args, kind.replace("-", "_")), 1):
            path = Path(value).absolute()
            source_id = f"{kind}:{index}"
            try:
                key = (kind, canonical(path))
            except (OSError, RuntimeError, ValueError):
                sources.append({"id": source_id, "kind": kind, "status": "path_unresolvable", "limits": []})
                continue
            if key in seen:
                sources.append({"id": source_id, "kind": kind, "status": "duplicate_input",
                                "same_as": seen[key], "limits": []})
                continue
            seen[key] = source_id
            if kind == "skill-root":
                source = collect_skills(path, source_id)
                for skill in source["skills"]:
                    if skill.get("name"):
                        names[skill["name"]].append(skill["id"])
            elif kind == "rollout":
                source = collect_rollout(path, source_id, args.max_rollout_mib * 1024 * 1024)
            else:
                source = collect_small(path, source_id, kind)
            sources.append(source)
    return {
        "schema_version": 1,
        "purpose": "explicit_input_metadata_inventory_not_diagnosis",
        "source_id_mapping": "kind_and_1_based_argument_order_no_absolute_paths",
        "sources": sources,
        "duplicate_skill_names": [{"name": name, "sources": ids, "meaning": "inspection_lead_only"}
                                  for name, ids in sorted(names.items()) if len(ids) > 1],
        "limits": ["no_automatic_discovery_or_chat_access", "no_instruction_or_conversation_content",
                   "not_proof_of_loading_routing_causation_or_improvement"],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "instruction", "skill-root", "rollout"):
        parser.add_argument(f"--{name}", action="append", default=[], metavar="PATH")
    parser.add_argument("--max-rollout-mib", type=int, choices=range(1, 65), default=8, metavar="1..64")
    parser.add_argument("--output", type=Path, help="New JSON file; omit for stdout. Never overwrites.")
    args = parser.parse_args(argv)
    count = sum(len(getattr(args, key)) for key in ("config", "instruction", "skill_root", "rollout"))
    if not 1 <= count <= 256:
        parser.error("Supply 1 to 256 explicit input paths.")
    try:
        if args.output:
            output = args.output.absolute()
            if linked(output) or output.exists() or not output.parent.is_dir():
                parser.error("Output must be a new file in an existing, non-linked directory.")
            for root in args.skill_root:
                if output.resolve().is_relative_to(Path(root).resolve()):
                    parser.error("Output must be outside each inventoried skill root.")
        report = build_report(args)
        encoded = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        if args.output:
            with args.output.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(encoded)
        else:
            sys.stdout.write(encoded)
    except (OSError, RuntimeError):
        parser.exit(2, "Could not read a path or create the report; no existing file was overwritten.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
