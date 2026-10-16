"""Synthetic file/CLI tests. These do not evaluate an Agent's judgment."""

from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

PACKAGE = Path(__file__).resolve().parents[1]
SCRIPT = PACKAGE / "scripts" / "collect_evidence.py"
spec = importlib.util.spec_from_file_location("collector", SCRIPT)
collector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collector)


class CollectorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def write(self, name, text):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def run_cli(self, *args):
        return subprocess.run([sys.executable, "-X", "utf8", str(SCRIPT), *map(str, args)],
                              capture_output=True, text=True, encoding="utf-8", timeout=20)

    def test_explicit_scope_and_private_content_are_not_exported(self):
        secret = "sk-EXAMPLE_PRIVATE_VALUE_DO_NOT_EXPORT"
        config = self.write("config.toml", f'model = "gpt-6-astra"\nmodel_reasoning_effort = "max"\n'
                            f'api_key = "{secret}"\n[model_providers.private]\nbase_url = "https://private.invalid"\n'
                            f'env_key = "{secret}"\n[memories]\nuse_memories = true\n')
        instruction = self.write("AGENTS.md", f"Keep current user scope. {secret}\n")
        # An unrelated malformed file must never become a source implicitly.
        self.write("other-chat.jsonl", "not json and not authorized")
        old = {p: sha256(p.read_bytes()).hexdigest() for p in (config, instruction)}
        report_path = self.root / "inventory.json"
        run = self.run_cli("--config", config, "--instruction", instruction, "--output", report_path)
        self.assertEqual(run.returncode, 0, run.stderr)
        serialized = report_path.read_text(encoding="utf-8")
        report = json.loads(serialized)
        self.assertEqual(len(report["sources"]), 2)
        fields = report["sources"][0]["stored_top_level_fields"]
        self.assertEqual(fields["model"], "gpt-6-astra")
        self.assertEqual(fields["memories"]["use_memories"], True)
        for value in (secret, "https://private.invalid", str(self.root), "Keep current user scope", "other-chat"):
            self.assertNotIn(value, serialized)
        self.assertEqual(old, {p: sha256(p.read_bytes()).hexdigest() for p in old})

    def test_partial_rollout_keeps_usage_semantics_and_hides_messages(self):
        rows = [
            {"type": "turn_context", "payload": {"model": "gpt-6-astra", "effort": "max", "cwd": "PRIVATE_PATH"}},
            {"type": "response_item", "payload": {"role": "user", "content": "PRIVATE_MESSAGE"}},
            {"type": "event_msg", "payload": {"type": "token_count", "info": {
                "last_token_usage": {"input_tokens": 100, "cached_input_tokens": 60},
                "total_token_usage": {"input_tokens": 900, "cached_input_tokens": 500},
                "model_context_window": 2000}}},
            {"type": "compacted", "payload": {"message": "PRIVATE_SUMMARY"}},
            {"type": "event_msg", "payload": {"type": "context_compacted"}},
            {"type": "future_schema", "payload": {"secret": "PRIVATE_UNKNOWN"}},
        ]
        path = self.write("events.jsonl", "\n".join(json.dumps(row) for row in rows) + "\nBROKEN JSON\n")
        result = collector.collect_rollout(path, "rollout:1", 1024 * 1024)
        self.assertEqual(result["status"], "partial")
        self.assertTrue(result["complete_file_read"])
        self.assertEqual(result["malformed_lines"], 1)
        self.assertEqual(result["unsupported_records"], 1)
        event = result["last_observed_token_event"]
        self.assertEqual(event["last_token_usage"]["input_tokens"], 100)
        self.assertEqual(event["total_token_usage"]["input_tokens"], 900)
        self.assertEqual(result["record_types"]["compacted"], 1)
        self.assertEqual(result["context_compacted_event_records"], 1)
        self.assertNotIn("PRIVATE", json.dumps(result))

    def test_latest_token_event_does_not_reuse_previous_usage_fields(self):
        path = self.write("events.jsonl", "\n".join(json.dumps({"type": "event_msg", "payload": {
            "type": "token_count", "info": info}}) for info in [
                {"last_token_usage": {"input_tokens": 100}},
                {"total_token_usage": {"input_tokens": 300, "output_tokens": True, "total_tokens": -2}},
            ]))
        event = collector.collect_rollout(path, "rollout:1", 1024)["last_observed_token_event"]
        self.assertEqual(event, {"line": 2, "total_token_usage": {"input_tokens": 300}})

    def test_prefix_and_oversized_line_limits_are_visible(self):
        row = json.dumps({"type": "turn_context", "payload": {"model": "gpt-6-sol"}}) + "\n"
        later = json.dumps({"type": "turn_context", "payload": {"model": "gpt-6-astra"}}) + "\n"
        path = self.write("long.jsonl", row + later)
        first_line_bytes = len(path.read_bytes().splitlines(keepends=True)[0])
        result = collector.collect_rollout(path, "rollout:1", first_line_bytes)
        self.assertFalse(result["complete_file_read"])
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["last_observed_turn_fields"]["model"], "gpt-6-sol")
        huge = self.write("huge.jsonl", "x" * (collector.LINE_LIMIT + 1))
        result = collector.collect_rollout(huge, "rollout:2", 3 * collector.LINE_LIMIT)
        self.assertEqual(result["lines_observed"], 0)
        self.assertFalse(result["complete_file_read"])

    def test_line_count_and_small_file_limits(self):
        path = self.write("lines.jsonl", '{}\n{}\n{}\n')
        with patch.object(collector, "LINE_COUNT_LIMIT", 2):
            result = collector.collect_rollout(path, "rollout:1", 1024)
        self.assertEqual(result["lines_observed"], 2)
        self.assertFalse(result["complete_file_read"])
        big = self.write("large.md", "z" * (collector.SMALL_LIMIT + 1))
        result = collector.collect_small(big, "instruction:1", "instruction")
        self.assertEqual(result["status"], "too_large")
        self.assertNotIn("sha256", result)

    def test_missing_invalid_and_unsupported_inputs_do_not_hide_valid_source(self):
        broken = self.write("broken.toml", 'key = "PRIVATE_UNCLOSED')
        invalid = self.root / "invalid.md"
        invalid.write_bytes(b"\xff\xfe\x81")
        valid = self.write("valid.md", "keep this\n")
        run = self.run_cli("--config", broken, "--config", self.root / "missing.toml",
                           "--instruction", invalid, "--instruction", valid)
        self.assertEqual(run.returncode, 0, run.stderr)
        statuses = [r["status"] for r in json.loads(run.stdout)["sources"]]
        self.assertEqual(statuses, ["invalid_toml", "missing", "invalid_utf8", "ok"])
        self.assertNotIn("PRIVATE", run.stdout + run.stderr)

    def test_skill_duplicates_are_only_inventory_leads_and_nested_dirs_are_not_scanned(self):
        content = '---\nname: same-name\ndescription: "Example"\n---\nPRIVATE_BODY\n'
        self.write("skills/a/SKILL.md", content)
        self.write("skills/b/SKILL.md", content)
        self.write("skills/work/deep/SKILL.md", content.replace("same-name", "nested-name"))
        run = self.run_cli("--skill-root", self.root / "skills", "--skill-root", self.root / "skills")
        self.assertEqual(run.returncode, 0, run.stderr)
        report = json.loads(run.stdout)
        self.assertEqual(len(report["sources"][0]["skills"]), 2)
        self.assertEqual([s["relative_path"] for s in report["sources"][0]["skills"]],
                         ["a/SKILL.md", "b/SKILL.md"])
        self.assertEqual(report["sources"][1]["status"], "duplicate_input")
        duplicate = report["duplicate_skill_names"][0]
        self.assertEqual(duplicate["meaning"], "inspection_lead_only")
        self.assertEqual(len(duplicate["sources"]), 2)
        self.assertNotIn("PRIVATE_BODY", run.stdout)
        self.assertNotIn("nested-name", run.stdout)

    def test_skill_metadata_parser_does_not_invent_names(self):
        for text in ('---\nname: "mismatch\'\ndescription: example\n---',
                     '---\nname: okay\nname: different\ndescription: example\n---',
                     '---\nname: |\n  multiline\ndescription: example\n---'):
            self.assertIsNone(collector.skill_metadata(text)["name"])

    def test_linked_input_is_not_followed(self):
        target = self.write("outside/AGENTS.md", "PRIVATE_OUTSIDE")
        link = self.root / "linked.md"
        try:
            link.symlink_to(target)
        except OSError:
            self.skipTest("Host does not allow creating a symlink; real symlink check not exercised.")
        result = collector.collect_small(link, "instruction:1", "instruction")
        self.assertEqual(result["status"], "linked_path_skipped")
        self.assertNotIn("bytes_read", result)

    def test_directory_entry_limit_is_reported(self):
        for index in range(3):
            self.write(f"skills/{index}/SKILL.md", f"---\nname: s{index}\ndescription: test\n---\n")
        with patch.object(collector, "ROOT_ENTRY_LIMIT", 2):
            result = collector.collect_skills(self.root / "skills", "skill-root:1")
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["entries_examined"], 2)

    def test_linked_directory_and_ancestor_are_not_followed(self):
        target = self.write("outside/hidden/SKILL.md", '---\nname: outside\ndescription: secret\n---\n')
        root = self.root / "skills"
        root.mkdir()
        link = root / "linked"
        try:
            link.symlink_to(target.parent, target_is_directory=True)
        except OSError:
            self.skipTest("Host does not allow creating a directory symlink.")
        result = collector.collect_skills(root, "skill-root:1")
        self.assertEqual(result["linked_entries_skipped"], 1)
        self.assertEqual(result["skills"], [])
        direct = collector.collect_small(link / "SKILL.md", "instruction:1", "instruction")
        self.assertEqual(direct["status"], "linked_path_skipped")

    def test_output_never_overwrites_inputs_or_existing_reports(self):
        source = self.write("AGENTS.md", "ORIGINAL")
        output = self.write("report.json", "EXISTING_REPORT")
        for target in (source, output):
            run = self.run_cli("--instruction", source, "--output", target)
            self.assertEqual(run.returncode, 2)
        self.assertEqual(source.read_text(), "ORIGINAL")
        self.assertEqual(output.read_text(), "EXISTING_REPORT")
        inside = self.root / "skills" / "new.json"
        inside.parent.mkdir()
        run = self.run_cli("--skill-root", inside.parent, "--output", inside)
        self.assertEqual(run.returncode, 2)
        self.assertFalse(inside.exists())

    def test_custom_model_fields_are_not_forwarded(self):
        path = self.write("config.toml", 'model = "https://private.invalid/token"\n'
                          'model_reasoning_effort = "PRIVATE_VALUE"\n'
                          'profile = "PRIVATE_PROFILE"\n[profiles.PRIVATE_PROFILE]\nmodel = "secret"\n')
        result = collector.collect_small(path, "config:1", "config")
        encoded = json.dumps(result)
        self.assertNotIn("PRIVATE", encoded)
        self.assertNotIn("private.invalid", encoded)
        self.assertIn("profile_overrides_not_resolved", result["limits"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
