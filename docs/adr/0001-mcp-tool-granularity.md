# ADR 0001: Separate MCP tools per operation, not one unified `edit_image`

**Status:** Proposed

## Context

The MCP server could expose one flexible `edit_image(operation, params)` tool,
or separate `generate_image` / `inpaint` / `upscale` / `score_image` tools.
This affects how reliably the agent picks the right action during the
critique loop, and how easy each tool is to test in isolation.

## Options considered

1. **One unified `edit_image` tool** — fewer tool definitions, but the model
   has to get the `operation` string and nested params right every call, and
   tool-selection ambiguity moves inside a single function instead of being
   resolved by MCP's own tool-choice mechanism.
2. **Separate tools per operation** — more tool definitions, but each has a
   narrow, unambiguous docstring the model can match against intent directly,
   and each is independently unit-testable via `/tdd`.

## Decision

Separate tools per operation (`generate_image`, `inpaint`, `upscale`,
`score_image`, `get_history`). Tool-selection accuracy matters more here than
tool-count — this is the crux of the "improvement loop" thesis, so it's worth
spending extra tool definitions on it.

## Consequences

Adding a new operation later (e.g. `outpaint`) means a new tool rather than a
new `operation` enum value — slightly more boilerplate, but it keeps each
tool's docstring and test file focused. Revisit if the tool count grows large
enough that the agent starts confusing similarly-named tools.
