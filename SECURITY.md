# Security and privacy

Do not commit API tokens, private service URLs, student videos, spreadsheets,
labels, raw model responses, or derived frames. The repository intentionally
requires the Qwen endpoint and token through environment variables.

Before publishing a fork, run the checks documented in `docs/privacy.md` and
inspect the complete Git index with `git ls-files`.

Report security issues privately to the repository owner instead of opening a
public issue containing credentials or personal data.
