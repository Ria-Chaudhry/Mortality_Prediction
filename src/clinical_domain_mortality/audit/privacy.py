"""Conservative text scan for material that must not enter public outputs."""

from __future__ import annotations

import re
from pathlib import Path

from ..errors import IntegrityError

SENSITIVE_PATTERNS = {
    "credential assignment": re.compile(
        r"(?i)(password|passwd|api[_-]?key|access[_-]?token|secret)\s*[:=]\s*[^\s'\"]{4,}"
    ),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "database URL with authority": re.compile(
        r"(?i)(?:postgres(?:ql)?|mysql|mssql|oracle)://[^/\s]+@"
    ),
}


def scan_public_tree(root: Path) -> None:
    findings = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() in {".pdf", ".parquet", ".gz"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for label, pattern in SENSITIVE_PATTERNS.items():
            if pattern.search(text):
                findings.append(f"{path.relative_to(root)}: {label}")
    if findings:
        raise IntegrityError(f"Public output privacy scan failed: {findings}")
