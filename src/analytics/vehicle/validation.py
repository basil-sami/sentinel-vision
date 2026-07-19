import re

_REGION_PATTERNS: dict[str, str] = {
    "us": r"^[A-Z0-9]{1,8}$",
    "uk": r"^[A-Z]{2}\d{2}[A-Z]{3}$",
    "eu": r"^[A-Z]{1,3}[-\s]?\d{1,4}[A-Z]{0,2}$",
    "generic": r"^[A-Z0-9]{3,8}$",
}

_CHARSET = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")


def clean_plate(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", text.upper().strip())


def validate_plate(text: str, min_len: int = 3, max_len: int = 9) -> tuple[str, float]:
    cleaned = clean_plate(text)
    if len(cleaned) < min_len or len(cleaned) > max_len:
        return ("", 0.0)

    valid_chars = sum(1 for c in cleaned if c in _CHARSET)
    ratio = valid_chars / len(cleaned) if cleaned else 0
    if ratio < 0.8:
        return ("", 0.0)

    for _, pattern in _REGION_PATTERNS.items():
        if re.match(pattern, cleaned):
            return (cleaned, round(ratio, 3))

    return (cleaned, round(ratio * 0.7, 3))
