# Security and privacy

Do not commit API tokens, private service URLs, student videos, spreadsheets,
labels, raw model responses, or derived frames. Workflow V2 and Agent require
model connection details through environment variables; Workflow V1 remains an
unchanged historical snapshot.

Before publishing a fork, run the checks documented in `workflow/v2/docs/privacy.md` and
inspect the complete Git index with `git ls-files`.

Report security issues privately to the repository owner instead of opening a
public issue containing credentials or personal data.
