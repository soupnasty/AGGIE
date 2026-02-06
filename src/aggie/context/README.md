# Context Management

> Aggie doesn't remember conversations — it maintains project state.

## Three-Tier Model

### Sacred Context
Decisions, agent definitions, tool schemas, constraints. **Never summarized.** Kept verbatim until explicitly revoked or the user clears context.

### Working Context
History summary + recent messages. Periodically compressed via a cheap Haiku call to preserve meaning while reducing tokens.

### Disposable Context
Pleasantries, resolved clarifications, abandoned approaches. Dropped during compression cycles.

## Modules

| File | Role |
|------|------|
| `project_state.py` | Pure data model — `ProjectState`, sacred dataclasses, `Turn` |
| `composer.py` | Prompt assembly — builds `(system_prompt, messages)` from state |
| `compressor.py` | Haiku-powered sacred extraction + working context compression |

## Compression Pipeline

Every **N turns** (default: 7), a fire-and-forget compression cycle runs:

1. **Extract** — Haiku scans recent messages for decisions, agents, tools, constraints
2. **Promote** — New sacred content added (deduplicated) to `ProjectState`
3. **Compress** — Older messages summarized into `history_summary`
4. **Trim** — `recent_messages` reduced to last N (default: 6)

Cost: ~$0.001 per cycle at Haiku pricing.

## Prompt Assembly Order

Stable content first to maximize prompt caching:

```
System Prompt (cached)     — Aggie's identity + behavior rules
Sacred Block (cached)      — Decisions, agents, tools, constraints
History Summary (fresh)    — Compressed narrative of past conversation
Recent Messages (fresh)    — Last N raw messages for coherence
```

## Context Clear

Users can clear all context (sacred + working) by saying one of:

- "Hey Aggie, fresh start"
- "Hey Aggie, new conversation"
- "Hey Aggie, forget everything"
- "Hey Aggie, start over"
- "Hey Aggie, self destruct"

Context can also be cleared via IPC: `aggie-ctl context-clear`

## Configuration

In `~/.config/aggie/config.yaml`:

```yaml
context:
  compress_every_turns: 7    # Compression interval
  recent_window: 6           # Messages kept after compression
  haiku_model: claude-haiku-4-5-20251001
  max_context_tokens: 8000
```
