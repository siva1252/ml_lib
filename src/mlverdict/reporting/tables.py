"""ASCII tables that stay readable in Windows terminals and logs."""

from __future__ import annotations

from collections.abc import Sequence


def ascii_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    *,
    aligns: Sequence[str] | None = None,
) -> str:
    if not headers:
        return ""
    n = len(headers)
    align = list(aligns) if aligns is not None else ["l"] * n
    if len(align) != n:
        raise ValueError("aligns must match header count")
    widths = [len(h) for h in headers]
    str_rows: list[list[str]] = []
    for row in rows:
        cells = [str(row[i]) if i < len(row) else "" for i in range(n)]
        str_rows.append(cells)
        for i, cell in enumerate(cells):
            widths[i] = max(widths[i], len(cell))

    def pad(text: str, width: int, how: str) -> str:
        if how == "r":
            return text.rjust(width)
        if how == "c":
            return text.center(width)
        return text.ljust(width)

    rule = "+-" + "-+-".join("-" * w for w in widths) + "-+"
    head = "| " + " | ".join(pad(headers[i], widths[i], align[i]) for i in range(n)) + " |"
    body = [
        "| " + " | ".join(pad(row[i], widths[i], align[i]) for i in range(n)) + " |"
        for row in str_rows
    ]
    if not body:
        return "\n".join([rule, head, rule, rule])
    return "\n".join([rule, head, rule, *body, rule])
