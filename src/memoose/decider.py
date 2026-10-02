"""The decider: memory's cheap decisions made by a local System One decision model.

A decision model does not generate. Given a state and a typed question it returns a calibrated
distribution over fixed options in one forward pass: a `choice`, a `score` on a rubric, or a
`noul` (the probability that a statement holds). Several of memoose's decisions have that shape
and are made today by a heuristic or a whole LLM call: is this transcript slice worth a capture,
does this step need the advisor now, which recall mode fits this query.

The model is Laya multilingual (mmBERT-base encoder plus a decision head, Apache-2.0), run from
its ONNX export with onnxruntime and tokenizers, on this machine, with no API key. `memoose
install` fetches it once (~0.65 GB, fp16); so does `memoose decide --fetch`.

`MEMOOSE_DECIDER` picks the mode:

- `on` (default): the decider's answer is used. When it cannot run (the model is not fetched),
  the caller's heuristic decides and the failure is recorded as an `error` row, so
  `memoose decide --report` shows it rather than it passing unseen.
- `shadow`: the decider runs next to the heuristic and is only recorded; the heuristic decides.
  Each record holds what the decider said and what actually happened: agreement per decision,
  and the labelled data a fine-tune would need.
- `off`: nothing here is loaded.

Only decisions the zero-shot model answered well in a probe are listed (docs/adr/0008-decider.md).
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
from functools import lru_cache
from pathlib import Path

from .store.datasets import data_dir

REPO = "onnx-community/laya-multilingual-ONNX"
FILES = ("config.json", "tokenizer.json", "tokenizer_config.json", "onnx/model_fp16.onnx", "onnx/model_fp16.onnx_data")
MODES = ("off", "shadow", "on")
WINDOW_CHARS = 3000  # ~1024 tokens is the model's whole sequence; longer text is judged in windows
MAX_WINDOWS = 4

# name -> the typed question, the option that means "yes" for a gate, and its threshold.
# The wording is the one that held up in the probe; rephrase only against `--report` numbers.
DECISIONS: dict[str, dict] = {
    "capture": {
        "t": "choice",
        "ins": "What kind of content is this exchange?",
        "crit": {
            "durable": "a decision, rule, preference, ownership, or a reason someone would need later",
            "transient": "a request to run something, a status report, or a description of a file",
        },
        "yes": "durable",
        "threshold": 0.5,
    },
    "advise": {
        "t": "noul",
        "ins": "Is the agent about to do something wrong, risky, or contrary to the user's instructions in this step?",
        "threshold": 0.5,
    },
    "route": {
        "t": "choice",
        "ins": "Which kind of memory search answers this question best?",
        "crit": {
            "facts": "a direct fact about a named thing: who owns it, what it uses, why it was chosen",
            "temporal": "when something happened, or what changed over time",
            "rules": "a standing rule, convention, or preference to follow",
            "neighbourhood": "how two or more things relate, connect, or depend on each other",
            "summaries": "an overview or summary of the whole project",
        },
    },
}


class DeciderError(RuntimeError):
    """The decider cannot run here: extra not installed, model not fetched, or a bad request."""


def mode() -> str:
    m = (os.environ.get("MEMOOSE_DECIDER") or "on").strip().lower()
    return m if m in MODES else "on"


def model_dir() -> Path:
    override = os.environ.get("MEMOOSE_DECIDER_MODEL")
    return Path(override).expanduser() if override else data_dir() / "models" / REPO.split("/")[1]


def log_path() -> Path:
    return data_dir() / "decisions.jsonl"


def fetch(progress=print) -> Path:
    """Download the ONNX bundle once; a file already present at its full size is kept."""
    root = model_dir()
    for name in FILES:
        dest = root / name
        url = f"https://huggingface.co/{REPO}/resolve/main/{name}"
        with urllib.request.urlopen(url) as r:
            size = int(r.headers.get("Content-Length") or 0)
            if dest.exists() and size and dest.stat().st_size == size:
                continue
            progress(f"fetching {name} ({size / 1e6:.0f} MB)")
            dest.parent.mkdir(parents=True, exist_ok=True)
            part = dest.with_suffix(dest.suffix + ".part")
            with open(part, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            part.replace(dest)
    return root


class Laya:
    """laya's `build_sequence` and one ONNX forward pass; matches `laya.Agent.predict`'s probabilities."""

    def __init__(self, root: Path) -> None:
        if not (root / FILES[-1]).exists():  # checked first: it is the common case, and costs no import
            raise DeciderError(f"no model at {root}; run `memoose decide --fetch` once")
        import numpy as np
        import onnxruntime as ort
        from tokenizers import Tokenizer
        self.np = np
        self.cfg = json.loads((root / "config.json").read_text())["laya"]
        special = json.loads((root / "tokenizer_config.json").read_text())
        self.tok = Tokenizer.from_file(str(root / "tokenizer.json"))
        vocab = self.tok.get_vocab()
        self.mask = special["mask_token"]
        self.cls, self.sep, self.mask_id = vocab[special["cls_token"]], vocab[special["sep_token"]], vocab[self.mask]
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        self.session = ort.InferenceSession(str(root / "onnx" / "model_fp16.onnx"), opts, providers=["CPUExecutionProvider"])

    def _ids(self, text: str) -> list[int]:
        return self.tok.encode(text.replace(self.mask, " "), add_special_tokens=False).ids

    @staticmethod
    def options(q: dict) -> list[str]:
        if q["t"] == "choice":
            return [f"{k}: {v}" if v else str(k) for k, v in q["crit"].items()]
        if q["t"] == "score":
            return [f"level {i}: {c}" for i, c in enumerate(q["crit"])]
        crit = q.get("crit") or {}
        return [f"false: {crit.get('false') or 'no, the statement does not hold'}",
                f"true: {crit.get('true') or 'yes, the statement holds'}"]

    def sequence(self, state: str, q: dict) -> tuple[list[int], list[int]]:
        """`[CLS] {type} question: {ins} [SEP] [MASK] opt … [SEP] state [SEP]`, laya's budgets."""
        head_max = self.cfg["head_max_len"]
        cap = self.cfg.get("max_option_tokens", 48)
        opts = [[self.mask_id] + self._ids(" " + o)[:cap] for o in self.options(q)]
        budget = head_max - sum(map(len, opts))
        if budget < 16:
            per = max(4, (head_max - 16) // len(opts))
            opts = [o[:per] for o in opts]
            budget = head_max - sum(map(len, opts))
        ids = [self.cls] + self._ids(f"{q['t']} question: {q['ins']}")[: max(8, budget)] + [self.sep]
        markers = []
        for o in opts:
            markers.append(len(ids))
            ids += o
        ids.append(self.sep)
        room = max(0, self.cfg["max_len"] - len(ids) - 1)
        return ids + self._ids(state)[:room] + [self.sep], markers

    def probabilities(self, states: list[str], q: dict) -> list[list[float]]:
        """One row of option probabilities per state, all states in one batch."""
        np = self.np
        seqs = [self.sequence(s, q) for s in states]
        b, length, k = len(seqs), max(len(s) for s, _ in seqs), len(seqs[0][1])
        ids, att = np.zeros((b, length), np.int64), np.zeros((b, length), np.int64)
        pos, live = np.zeros((b, k), np.int64), np.ones((b, k), bool)
        for i, (s, m) in enumerate(seqs):
            ids[i, : len(s)], att[i, : len(s)], pos[i] = s, 1, m
        qtype = self.cfg["qtypes"][q["t"]]
        logits, _ = self.session.run(None, {"input_ids": ids, "attention_mask": att, "marker_pos": pos,
                                            "marker_mask": live, "qtype": np.full(b, qtype, np.int64)})
        lo, hi = self.cfg.get("temperature_clamp", [0.5, 5.0])
        temp = min(hi, max(lo, self.cfg["temperature"][qtype]))
        z = logits[:, :k] / temp
        p = np.exp(z - z.max(axis=1, keepdims=True))
        return (p / p.sum(axis=1, keepdims=True)).tolist()


@lru_cache(maxsize=1)
def model() -> Laya:
    return Laya(model_dir())


def windows(text: str) -> list[str]:
    """Up to MAX_WINDOWS slices of WINDOW_CHARS, newest last; the rest of a long text is not judged."""
    text = text.strip()
    parts = [text[i : i + WINDOW_CHARS] for i in range(0, len(text), WINDOW_CHARS)] or [""]
    return parts[-MAX_WINDOWS:]  # ponytail: the newest windows only; a long slice's start goes unjudged


def decide(name: str, state: str) -> dict:
    """The decider's answer for `name` on `state`. A long state is judged per window, highest wins.

    Gates (`capture`, `advise`) answer `{"p", "yes"}`; `route` answers `{"choice", "probabilities"}`.
    """
    q = DECISIONS.get(name)
    if q is None:
        raise DeciderError(f"unknown decision {name!r}; one of: {', '.join(DECISIONS)}")
    rows = model().probabilities(windows(state), q)
    if q["t"] == "noul":
        p = max(r[1] for r in rows)
    elif "yes" in q:
        at = list(q["crit"]).index(q["yes"])
        p = max(r[at] for r in rows)
    else:
        keys = list(q["crit"])
        best = max(rows, key=max)
        return {"choice": keys[best.index(max(best))], "probabilities": {k: round(v, 4) for k, v in zip(keys, best)}}
    return {"p": round(p, 4), "yes": p >= q["threshold"]}


def record(name: str, state: str, answer: dict | None, actual=None, error: str | None = None) -> None:
    """Append one decision to the log. Never raises: the log must not break what it observes."""
    row = {"at": int(time.time()), "decision": name, "mode": mode(), "answer": answer, "actual": actual,
           "error": error, "state": state[:2000]}
    try:
        path = log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except OSError:
        pass


def consult(name: str, state: str, actual=None) -> dict | None:
    """Decide and record. None when the mode is off or the decider failed (the failure is recorded)."""
    if mode() == "off":
        return None
    try:
        answer = decide(name, state)
    except Exception as e:  # noqa: BLE001 - any failure means "heuristic decides", and is recorded
        record(name, state, None, actual, error=f"{type(e).__name__}: {e}")
        return None
    record(name, state, answer, actual)
    return answer


def report(path: Path | None = None) -> dict:
    """Per decision: how often the decider ran, failed, and agreed with what actually happened.

    `actual` is what memoose did. For `capture` that is the keeper's verdict (it stored something
    or replied NOTHING) and for `advise` whether the advisor emitted advice; both are judgments a
    larger model made, so agreement there is accuracy against a teacher. For `route` the actual
    is the regex router, which is a baseline, not ground truth.
    """
    out: dict[str, dict] = {}
    try:
        lines = (path or log_path()).read_text(encoding="utf-8").splitlines()
    except OSError:
        lines = []
    for line in lines:
        try:
            row = json.loads(line)
        except ValueError:
            continue
        s = out.setdefault(row["decision"], {"rows": 0, "errors": 0, "labelled": 0, "agree": 0, "last_error": None})
        s["rows"] += 1
        if row.get("error"):
            s["errors"] += 1
            s["last_error"] = row["error"]
            continue
        actual, answer = row.get("actual"), row.get("answer") or {}
        if actual is None:
            continue
        said = answer.get("yes", answer.get("choice"))
        s["labelled"] += 1
        s["agree"] += said == actual
    for s in out.values():
        s["agreement"] = round(s["agree"] / s["labelled"], 3) if s["labelled"] else None
    return {"mode": mode(), "model": str(model_dir()), "log": str(path or log_path()), "decisions": out}
