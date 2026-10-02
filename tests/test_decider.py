"""The decider: modes, the decision log, the router, and the two hook gates it drives."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from memoose import decider
from memoose.graph.retrieval import Router

ROOT = Path(__file__).resolve().parents[1]
HOOKS = ROOT / "harness" / "hooks"
sys.path.insert(0, str(HOOKS))
SWITCHES = ("MEMOOSE_HINTS", "MEMOOSE_AUTO_RECALL", "MEMOOSE_AUTO_CAPTURE", "MEMOOSE_AUTO_MAINTAIN", "MEMOOSE_ADVISOR")


def _hook(script: str, event: dict, env: dict) -> subprocess.CompletedProcess:
    base = {k: v for k, v in os.environ.items() if k not in SWITCHES}  # the switches default on
    return subprocess.run([sys.executable, str(HOOKS / script)], input=json.dumps(event),
                          capture_output=True, text=True, env={**base, **env}, timeout=60)


def _bin(tmp_path: Path, answer: dict) -> tuple[Path, Path, Path]:
    """A fake `memoose` that answers every decision with `answer`, and a `claude` that logs its calls."""
    fake = tmp_path / "bin"
    fake.mkdir()
    decides, claude = tmp_path / "decide-argv", tmp_path / "claude-argv"
    (fake / "memoose").write_text(f"#!/bin/sh\necho \"$@\" >> {decides}\ncat > /dev/null\necho '{json.dumps(answer)}'\n")
    (fake / "claude").write_text(f"#!/bin/sh\necho \"$@\" >> {claude}\ncat > /dev/null\necho NOTHING\n")
    for f in fake.iterdir():
        f.chmod(0o755)
    return fake, decides, claude


def _transcript(tmp_path: Path, rows: list[dict]) -> Path:
    t = tmp_path / "t.jsonl"
    t.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return t


# ----- the module ----------------------------------------------------------------------------
def test_on_by_default_and_a_missing_model_is_recorded_not_raised(tmp_path, monkeypatch):
    monkeypatch.delenv("MEMOOSE_DECIDER")
    monkeypatch.setenv("MEMOOSE_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("MEMOOSE_DECIDER_MODEL", str(tmp_path / "nowhere"))
    decider.model.cache_clear()
    assert decider.mode() == "on"
    assert decider.consult("capture", "We moved billing to Postgres.") is None, "the heuristic decides"
    s = decider.report()["decisions"]["capture"]
    assert s["errors"] == 1 and "decide --fetch" in s["last_error"], "the failure is visible in --report"


def test_report_scores_agreement_against_what_happened(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMOOSE_DATA_DIR", str(tmp_path))
    decider.record("advise", "s", {"p": 0.9, "yes": True}, actual=True)
    decider.record("advise", "s", {"p": 0.9, "yes": True}, actual=False)
    decider.record("advise", "s", {"p": 0.1, "yes": False})  # no label: counted, not scored
    decider.record("route", "q", {"choice": "temporal", "probabilities": {}}, actual="temporal")
    d = decider.report()["decisions"]
    assert (d["advise"]["rows"], d["advise"]["labelled"], d["advise"]["agreement"]) == (3, 2, 0.5)
    assert d["route"]["agreement"] == 1.0


def test_windows_keep_the_newest_text():
    text = "a" * decider.WINDOW_CHARS * 6 + "LATEST"
    parts = decider.windows(text)
    assert len(parts) == decider.MAX_WINDOWS and parts[-1].endswith("LATEST")


# ----- the router ----------------------------------------------------------------------------
def test_router_takes_the_decider_when_on_and_the_rules_otherwise(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMOOSE_DATA_DIR", str(tmp_path))
    asked = []
    monkeypatch.setattr(decider, "decide", lambda name, state: asked.append(state) or
                        {"choice": "temporal", "probabilities": {"temporal": 0.9, "facts": 0.1}})
    router = Router()
    monkeypatch.setenv("MEMOOSE_DECIDER", "on")
    r = router.route("which gateway does billing use")
    assert (r.mode, r.runner_up, r.matched) == ("temporal", "facts", ["decider"])
    assert router.route('"exact phrase"').mode == "lexical" and router.route("what are the rules").mode == "rules"
    assert len(asked) == 1, "a query the rules can read never loads the model"
    monkeypatch.setattr(decider, "decide", lambda name, state: {"choice": "temporal", "probabilities": {"temporal": 0.4, "facts": 0.35}})
    assert router.route("which gateway does billing use").mode == "hybrid", "an unsure answer keeps the default"
    monkeypatch.setenv("MEMOOSE_DECIDER", "shadow")
    assert router.route("which gateway does billing use").mode == "hybrid", "shadow: the rules decide"
    rows = [json.loads(line) for line in decider.log_path().read_text().splitlines()]
    assert rows[-1]["actual"] == "hybrid" and rows[-1]["answer"]["choice"] == "temporal"


# ----- the hook gates ------------------------------------------------------------------------
LONG = "We decided to move authentication to JWT because sessions did not scale. " * 12


def test_capture_gate_skips_the_keeper_when_the_decider_says_transient(tmp_path):
    fake, decides, claude = _bin(tmp_path, {"p": 0.1, "yes": False})
    t = _transcript(tmp_path, [{"type": "user", "message": {"role": "user", "content": LONG}}])
    env = {"MEMOOSE_DATA_DIR": str(tmp_path / "data"), "PATH": f"{fake}:{os.environ['PATH']}", "MEMOOSE_DECIDER": "on"}
    assert _hook("capture.py", {"session_id": "s", "cwd": str(tmp_path), "transcript_path": str(t)}, env).returncode == 0
    assert "decide capture" in decides.read_text()
    assert not claude.exists(), "a slice the decider calls transient never reaches the keeper"


def test_capture_shadow_runs_the_keeper_and_labels_the_decision(tmp_path):
    fake, decides, claude = _bin(tmp_path, {"p": 0.9, "yes": True})
    t = _transcript(tmp_path, [{"type": "user", "message": {"role": "user", "content": LONG}}])
    env = {"MEMOOSE_DATA_DIR": str(tmp_path / "data"), "PATH": f"{fake}:{os.environ['PATH']}", "MEMOOSE_DECIDER": "shadow"}
    assert _hook("capture.py", {"session_id": "s", "cwd": str(tmp_path), "transcript_path": str(t)}, env).returncode == 0
    assert claude.exists(), "shadow never gates"
    assert "decide capture --actual transient" in decides.read_text(), "the keeper's NOTHING is the label"


def test_advisor_defers_a_safe_step_to_the_end_of_turn_review(tmp_path):
    fake, decides, claude = _bin(tmp_path, {"p": 0.05, "yes": False})
    t = _transcript(tmp_path, [
        {"type": "user", "message": {"content": "Fix the typo in README."}},
        {"type": "assistant", "message": {"content": [{"type": "tool_use", "name": "Edit", "input": {"file_path": "README.md"}}]}},
    ])
    env = {"MEMOOSE_DATA_DIR": str(tmp_path / "data"), "PATH": f"{fake}:{os.environ['PATH']}", "MEMOOSE_DECIDER": "on"}
    ev = {"session_id": "s", "cwd": str(tmp_path), "transcript_path": str(t), "hook_event_name": "PostToolUse"}
    assert _hook("advisor.py", ev, env).returncode == 0, "a deferral must not loop on its own unread lines"
    assert "decide advise" in decides.read_text() and not claude.exists()
    with t.open("a") as f:
        f.write(json.dumps({"type": "assistant", "message": {"content": "Fixed."}}) + "\n")
    assert _hook("advisor.py", {**ev, "hook_event_name": "Stop"}, env).returncode == 0
    assert claude.exists(), "the end-of-turn review is never gated"
    assert decides.read_text().count("decide advise") == 1


def test_advisor_reviews_a_step_the_decider_calls_risky(tmp_path):
    fake, _, claude = _bin(tmp_path, {"p": 0.97, "yes": True})
    t = _transcript(tmp_path, [
        {"type": "user", "message": {"content": "Never touch production."}},
        {"type": "assistant", "message": {"content": [{"type": "tool_use", "name": "Bash", "input": {"command": "psql $PROD -c 'DROP TABLE users'"}}]}},
    ])
    env = {"MEMOOSE_DATA_DIR": str(tmp_path / "data"), "PATH": f"{fake}:{os.environ['PATH']}", "MEMOOSE_DECIDER": "on"}
    ev = {"session_id": "s", "cwd": str(tmp_path), "transcript_path": str(t), "hook_event_name": "PostToolUse"}
    assert _hook("advisor.py", ev, env).returncode == 0
    assert claude.exists()


# ----- the real model, when this machine has it ----------------------------------------------
@pytest.mark.skipif(not (decider.model_dir() / decider.FILES[-1]).exists(), reason="model not fetched (memoose decide --fetch)")
def test_the_model_answers_the_probe_cases():
    decider.model.cache_clear()
    assert decider.decide("advise", "User: do not touch the production database.\n"
                                    "Agent: running psql $PROD_URL -c 'DROP TABLE users'")["yes"]
    assert not decider.decide("advise", "User: fix the typo in README.\nAgent: Edit README.md: 'teh' -> 'the'.")["yes"]
    assert decider.decide("route", "When did we switch the payment gateway?")["choice"] == "temporal"
    assert decider.decide("route", "What are the rules for writing tests in this repo?")["choice"] == "rules"
    assert not decider.decide("capture", "User: run the tests again\nAssistant: I ran pytest, 42 passed.")["yes"]
