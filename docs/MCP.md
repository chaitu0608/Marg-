# Marg MCP analyst

Marg's optional MCP server gives an MCP-compatible agent a read-only view of the saved evaluation evidence. It can retrieve allowlisted result JSON files, inspect the nearest saved fraud sensitivity case, and describe the model/router/investigator/gateway boundaries.

The tools do not accept arbitrary paths, execute shell commands, edit artifacts, score transactions, change decisions, or dispatch payments. The LLM can interpret the measurements and draft an explanation; the fraud model and router remain the decision owners.

## Install and run

```sh
python -m pip install -e '.[mcp]'
make mcp
```

The server uses MCP stdio, so configure it in a compatible client rather than opening a network port. Codex supports MCP servers through its CLI and shared IDE configuration. Register Marg with the installed entry point, replacing the absolute path with this checkout's location:

```sh
codex mcp add marg -- /absolute/path/to/Marg/.venv/bin/marg-mcp
codex mcp list
```

After the server appears in the list, start a new Codex session and select the Marg tools. Available tools are `get_marg_result`, `compare_fraud_policy`, and `marg_guardrails`. Result tools return saved JSON as evidence; they do not fabricate a missing artifact. Generate all saved results first with `make verify` in an environment that has `data/creditcard.csv`.

## DevOps gate

GitHub Actions runs pytest, Ruff, the README-versus-JSON consistency check, and a real stdio MCP smoke test on pushes and pull requests. The real dataset remains local and ignored, so the GitHub workflow does not run the full data-dependent benchmark. Run `make verify` locally when regenerating results.
