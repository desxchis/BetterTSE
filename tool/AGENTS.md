# Tool Module AGENTS

## Scope

These rules apply to everything under `tool/`.

## Why this area is high risk

`tool/` sits at the execution boundary between:

- LLM-produced plans
- deterministic tool implementations
- TEdit model integration
- metric-sensitive edited outputs

Small changes here can silently affect region boundaries, background preservation, and benchmark scores.

## Hard rules

- Do not modify `tool/` unless the task clearly belongs to tool execution, region editing, describers, composers, or the TEdit wrapper.
- Prefer fixing prompt parsing or planner logic in `agent/` or `modules/llm.py` when the bug is not actually in tool execution.
- Prefer fixing wrappers and adapters over editing `TEdit-main/`.
- Do not silently change public tool names, plan schema keys, or return-value shapes.
- Do not introduce hidden randomness into deterministic tool code.
- Do not broaden an edited region beyond the requested interval without explicit justification.

## Implementation rules

- Preserve outside-region values unless the tool is explicitly designed to alter them.
- Keep numeric behavior explicit: note clipping, smoothing, mask construction, and blend coefficients.
- When changing boundary handling, inspect off-by-one behavior and start/end inclusivity carefully.
- Keep CPU-safe behavior unless the task explicitly targets GPU-only execution.
- If a tool falls back from TEdit to math-only behavior, keep the fallback visible in logs or returned metadata.

## Validation rules

Any change in `tool/` should trigger, as applicable:

1. `python -m py_compile tool/*.py`
2. a targeted smoke test that executes the changed tool path
3. a background-preservation check when region editing is involved
4. a broader pipeline smoke test if plan-to-tool execution changed

## Done when

A `tool/` change is complete only when:

- the owning execution path was validated
- boundary behavior was checked
- return schema compatibility was preserved or intentionally updated
- any risk to editability / preservability / localization was called out in the final summary
