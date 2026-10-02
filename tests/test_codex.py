"""Codex as a second host: its transcript, its hook registration, and its headless runs."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOOKS = ROOT / "harness" / "hooks"
sys.path.insert(0, str(HOOKS))
import _common  # noqa: E402

from memoose.cli import integrations  # noqa: E402


def _rollout(tmp_path, items) -> Path:
    """Lines in the shape `codex exec` writes to ~/.codex/sessions/**/rollout-*.jsonl."""
    t = tmp_path / "rollout.jsonl"
    t.write_text("".join(json.dumps({"timestamp": "t", "type": k, "payload": p}) + "\n" for k, p in items))
    return t


ROLLOUT = [
    ("session_meta", {"id": "s"}),
    ("response_item", {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "<permissions>"}]}),
    ("response_item", {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "<environment_context>cwd</environment_context>"}]}),
    ("response_item", {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Add tenant scoping to the billing query."}]}),
    ("response_item", {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": '<advisory severity="nit">x</advisory>'}]}),
    ("response_item", {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Editing db.py."}]}),
    ("response_item", {"type": "custom_tool_call", "name": "exec", "input": "rg tenant db.py"}),
    ("event_msg", {"type": "agent_message", "message": "Editing db.py."}),
    ("response_item", {"type": "custom_tool_call_output", "output": [{"type": "input_text", "text": "db.py:4: tenant_id"}]}),
]


def test_rollout_lines_read_as_speech_and_tool_activity(tmp_path):
    import advisor
    t = _rollout(tmp_path, ROLLOUT)
    delta, end, mid_turn = advisor.read_delta(str(t), 0)
    assert "User: Add tenant scoping" in delta and "Agent: Editing db.py." in delta
    assert "Tool call exec" in delta and "tenant_id" in delta
    assert "environment_context" not in delta and "permissions" not in delta, "host context is not speech"
    assert "advisory" not in delta, "hook output comes back as a developer message and is dropped"
    assert delta.count("Editing db.py.") == 1, "event_msg duplicates are not read twice"
    assert end == len(ROLLOUT) and mid_turn
    text, n, tools = _common.read_exchange(str(t))
    assert "Add tenant scoping" in text and "Editing db.py." in text and tools and n == len(ROLLOUT)


def test_host_comes_from_the_hook_command():
    code = "import sys; sys.path.insert(0, %r); import _common; print(_common.host())" % str(HOOKS)
    run = lambda *a: subprocess.run([sys.executable, "-c", code, *a], capture_output=True, text=True).stdout.strip()  # noqa: E731
    assert run() == "claude" and run("--host", "codex") == "codex" and run("--host", "nope") == "claude"


def test_codex_install_registers_detached_hooks_beside_others(tmp_path, monkeypatch):
    monkeypatch.setattr(integrations.Path, "home", classmethod(lambda cls: tmp_path))
    codex_home = tmp_path / "codexhome"
    codex_home.mkdir()
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    (codex_home / "hooks.json").write_text(json.dumps({"hooks": {"Stop": [{"hooks": [{"type": "command", "command": "echo other"}]}]}}))

    res = integrations.install("codex")
    assert "/hooks" in res["note"], "the user is told Codex asks them to trust new hooks"
    hooks = json.loads((codex_home / "hooks.json").read_text())["hooks"]
    commands = [h for groups in hooks.values() for g in groups for h in g["hooks"]]
    ours = [h for h in commands if integrations.HOOK_MARK in h["command"]]
    assert any(h["command"] == "echo other" for h in commands), "someone else's hook survives"
    assert ours and all(h["command"].endswith("--host codex") for h in ours)
    assert not any("async" in h or "asyncRewake" in h for h in ours), "Codex would cancel them; they detach"
    assert set(hooks) >= {"SessionStart", "UserPromptSubmit", "PostToolUse", "Stop", "PreCompact"}
    assert (codex_home / "memoose" / "hooks" / "advisor.py").exists()
    assert integrations.status()["codex"]["hooks"] is True

    integrations.install("codex")
    again = json.loads((codex_home / "hooks.json").read_text())["hooks"]
    assert sum(integrations.HOOK_MARK in h["command"] for g in again.values() for x in g for h in x["hooks"]) == len(ours)

    assert integrations.uninstall("codex")["hooks_removed"] is True
    left = json.loads((codex_home / "hooks.json").read_text())["hooks"]
    assert left == {"Stop": [{"hooks": [{"type": "command", "command": "echo other"}]}]}


def _fake_codex(tmp_path) -> tuple[Path, Path]:
    """A `codex` that logs its argv, starts or resumes a thread, and answers per the schema."""
    fake_bin, log = tmp_path / "bin", tmp_path / "codex-argv"
    fake_bin.mkdir()
    (fake_bin / "codex").write_text(f"""#!{sys.executable}
import json, sys
args = sys.argv[1:]
open({str(log)!r}, "a").write(json.dumps(args) + "\\n")
sys.stdin.read()
out = args[args.index("-o") + 1]
json.dump({{"advisories": [{{"severity": "concern", "note": "The query drops tenant_id."}}]}}, open(out, "w"))
print(json.dumps({{"type": "thread.started", "thread_id": "thread-1"}}))
""")
    (fake_bin / "codex").chmod(0o755)
    return fake_bin, log


def _wait_for(pred, timeout=20.0):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.1)
    return False


def test_codex_advisor_detaches_resumes_its_thread_and_guards_each_note(tmp_path, monkeypatch):
    fake_bin, log = _fake_codex(tmp_path)
    t = tmp_path / "rollout.jsonl"
    t.write_text("")
    env = {**os.environ, "MEMOOSE_DATA_DIR": str(tmp_path / "data"), "PATH": f"{fake_bin}:{os.environ['PATH']}"}
    ev = {"session_id": "cx", "cwd": str(tmp_path), "transcript_path": str(t), "hook_event_name": "PostToolUse"}

    def step(text):
        with t.open("a") as f:
            f.write(json.dumps({"type": "response_item", "payload": {"type": "message", "role": "user",
                                "content": [{"type": "input_text", "text": text}]}}) + "\n")
        start = time.time()
        r = subprocess.run([sys.executable, str(HOOKS / "advisor.py"), "--host", "codex"], input=json.dumps(ev),
                           capture_output=True, text=True, env=env, timeout=30)
        assert r.returncode == 0 and time.time() - start < 5, "the hook returns at once; the review runs detached"

    step("Refactor the billing query.")
    assert _wait_for(lambda: log.exists() and len(log.read_text().splitlines()) == 1)
    step("Now also drop the tenant filter.")
    assert _wait_for(lambda: len(log.read_text().splitlines()) == 2)
    first, second = (json.loads(x) for x in log.read_text().splitlines())
    assert first[:1] == ["exec"] and "resume" not in first
    assert second[:3] == ["exec", "resume", "thread-1"], "the second review continues the same thread"
    assert "--output-schema" in first and 'sandbox_mode="workspace-write"' in first and "mcp_servers={}" in first

    monkeypatch.setenv("MEMOOSE_DATA_DIR", str(tmp_path / "data"))
    import advise
    with advise.session_state("cx") as st:
        notes = [a["note"] for a in st.get("mail", [])]
    assert notes == ["The query drops tenant_id."], "the repeat in the second review is suppressed by the guard"
