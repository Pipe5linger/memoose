---
status: accepted
extends: ADR 0003 (hooks), ADR 0006 (advisor); pattern from Tencent/teamai-cli (one set of hooks, per-host adapters)
---

# The hooks run on Codex as well as Claude Code, through one small host seam

Capture, hints and the Advisor were written against Claude Code. Codex has hooks with Claude's
JSON shape and event names, so most of the mechanism already fits. We checked each piece against
Codex 0.154 in a real session and adapted what differs rather than forking the scripts.
teamai-cli uses the same shape: every host runs the same hooks, and a per-host table maps events,
payloads and output.

## What we verified in Codex

- **Context injection.** `UserPromptSubmit` and `PostToolUse` accept `additionalContext`, and a
  PostToolUse note reaches the model before its next step in the same turn.
- **Stop.** `{"decision":"block","reason":…}` forces a continuation, and `stop_hook_active`
  guards against loops.
- **Transcript.** The hook payload has `session_id`, `cwd` and `transcript_path`. The transcript
  is a rollout JSONL of `response_item` lines. Tool calls are items of their own, and injected
  context, including our own Advisories, comes back as `developer` messages.
- **Async hooks.** They are cancelled when the session ends. For `codex exec` that is right after
  the last Stop, which is the capture that matters most. They cannot wake an idle session.
- **Trust.** A new hook runs only after the user trusts it in `/hooks`.
- **Headless runs.** `codex exec` takes a prompt on stdin, and `codex exec resume <thread>`
  continues a thread by the `thread_id` its `--json` output reports. There is no system-prompt
  flag and no per-command allowlist. `--output-schema` constrains the final answer.
- **Sandbox.** In the read-only sandbox SQLite cannot open memory. With `workspace-write` and the
  memory directory as the only writable root, recall works and writing the project is refused.

## Decision

1. **`--host <id>` on the hook command** says which client ran the script. With no flag it is
   Claude Code, whose plugin hooks carry none. `memoose install codex` writes the same hooks into
   `$CODEX_HOME/hooks.json` with the flag added.
2. **One transcript reader.** `transcript_message` turns a Claude line or a Codex rollout line
   into the same role and content blocks. It drops `developer` messages and tag-wrapped user text,
   which the host injected and the person never typed.
3. **Detach where the host kills async hooks.** On Codex the background hooks run synchronously,
   re-launch themselves detached (`--bg`) and return at once, so capture survives the session.
   On Claude Code they stay `async`, because `asyncRewake` needs the hook's own exit code.
4. **The headless runner per host.**
   - Claude Code: `claude -p` with an allowlist.
   - Codex: `codex exec` with `workspace-write`, memory as the only writable root, no MCP servers
     and low reasoning effort.
   - The Codex Advisor gets its policy in the first prompt of its thread, and answers in JSON
     against a schema. The runner puts each note through the same emission guard.
5. **Delivery is unchanged**, because Codex takes the same output. A late blocker cannot wake an
   idle Codex session, so it waits for the next prompt.

## Consequences

- A new host needs the same four things: its name in `HOSTS`, its transcript lines in
  `transcript_message`, its runner, and its hook file in the installer. These are where Gemini CLI,
  Cursor and OpenCode would go:
  - **Gemini CLI:** has the events (BeforeAgent, AfterTool, AfterAgent `decision:"deny"`), but its
    hooks are synchronous only, so it would detach.
  - **Cursor:** cannot inject at the prompt, and uses `followup_message` at stop.
  - **OpenCode:** takes a TypeScript plugin rather than shell hooks.

  None of them is installed where this was built, so none ships untested.
- The Codex Advisor can run any read-only shell command, not only recall. The sandbox, not an
  allowlist, is what keeps it from writing the project.
- With no model set, Codex's background jobs use the user's configured model at low reasoning
  effort. Set `MEMOOSE_ADVISOR_MODEL` and `MEMOOSE_CAPTURE_MODEL` to pin a smaller one.
