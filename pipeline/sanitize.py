"""
Strip secrets from episodes/*.md before they're sent to NIM.

Run BEFORE .\\run.bat. Rewrites files in place. Prints a summary of redactions.

Patterns redacted:
  - API keys: nvapi-, sk-, sk_test_, sk_live_, AKIA, ASIA, ghp_, gho_, ghs_,
    Bearer tokens, hugging face hf_, OpenAI proj keys, Anthropic ant_, etc.
  - .env-style key/value lines:  *_KEY=..., *_TOKEN=..., *_SECRET=..., PASSWORD=..., DSN=...
  - Generic long opaque strings adjacent to "key"/"token"/"secret" labels
"""
import re
import sys
from pathlib import Path

EPISODES_DIR = Path(__file__).resolve().parent.parent / "episodes"
MARKER = "[REDACTED]"

PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("nvidia",       re.compile(r"nvapi-[A-Za-z0-9_\-]{20,}")),
    ("openai",       re.compile(r"sk-(?:proj-)?[A-Za-z0-9_\-]{20,}")),
    ("openai-test",  re.compile(r"sk_(?:live|test)_[A-Za-z0-9]{20,}")),
    ("anthropic",    re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}")),
    ("aws-access",   re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("github",       re.compile(r"\b(?:ghp|gho|ghs|ghu|ghr)_[A-Za-z0-9]{30,}\b")),
    ("hf",           re.compile(r"\bhf_[A-Za-z0-9]{30,}\b")),
    ("google",       re.compile(r"\bAIza[0-9A-Za-z_\-]{30,}\b")),
    ("slack",        re.compile(r"\bxox[abprs]-[A-Za-z0-9\-]{20,}\b")),
    ("bearer",       re.compile(r"(?i)bearer\s+[A-Za-z0-9_\-\.=]{20,}")),
    ("basic-auth",   re.compile(r"(?i)basic\s+[A-Za-z0-9+/=]{20,}")),
    ("env-secret",   re.compile(
        r"(?im)^\s*([A-Z][A-Z0-9_]*?(?:KEY|TOKEN|SECRET|PASSWORD|PASSWD|PWD|DSN|CRED|API))\s*=\s*\S+"
    )),
    ("inline-secret", re.compile(
        r"(?i)\b(api[_-]?key|access[_-]?token|auth[_-]?token|client[_-]?secret|"
        r"private[_-]?key|password)\s*[:=]\s*['\"]?[A-Za-z0-9_\-\.]{16,}['\"]?"
    )),
]


def sanitize(text: str) -> tuple[str, dict[str, int]]:
    counts: dict[str, int] = {}
    for label, pat in PATTERNS:
        if label == "env-secret":
            new_text, n = pat.subn(lambda m: f"{m.group(1)}={MARKER}", text)
        elif label == "inline-secret":
            new_text, n = pat.subn(
                lambda m: re.sub(r"['\"]?[A-Za-z0-9_\-\.]{16,}['\"]?$", MARKER, m.group(0)),
                text,
            )
        else:
            new_text, n = pat.subn(MARKER, text)
        if n:
            counts[label] = n
        text = new_text
    return text, counts


def main() -> None:
    files = sorted(EPISODES_DIR.glob("*.md")) + sorted(EPISODES_DIR.glob("*.txt"))
    if not files:
        print(f"No episodes in {EPISODES_DIR}")
        return

    grand_total: dict[str, int] = {}
    for path in files:
        original = path.read_text(encoding="utf-8")
        cleaned, counts = sanitize(original)
        if not counts:
            print(f"  clean: {path.name}")
            continue
        path.write_text(cleaned, encoding="utf-8")
        summary = ", ".join(f"{k}={v}" for k, v in counts.items())
        print(f"  redacted: {path.name}  [{summary}]")
        for k, v in counts.items():
            grand_total[k] = grand_total.get(k, 0) + v

    print(f"\nDone. Files scanned: {len(files)}")
    if grand_total:
        print("Total redactions:")
        for k, v in sorted(grand_total.items(), key=lambda kv: -kv[1]):
            print(f"  {k}: {v}")
    else:
        print("Nothing matched.")


if __name__ == "__main__":
    main()
