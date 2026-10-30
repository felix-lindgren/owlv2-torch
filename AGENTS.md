# Repository agent guidance

- Do not add, propose, or recommend automated tests unless the user explicitly requests them.
- Prefer lightweight syntax, import, or manual smoke checks when verification is needed.
- Use `svm_backend="torch"` for new FLAME experiments; reserve sklearn for explicit reference comparisons and legacy checkpoint compatibility.
