from __future__ import annotations

import importlib


REQUIRED = ["numpy", "pandas", "yaml", "jsonschema", "sklearn"]


def main() -> None:
    missing = [name for name in REQUIRED if importlib.util.find_spec(name) is None]
    if missing:
        raise SystemExit(f"Missing packages: {', '.join(missing)}")
    print("Environment OK")


if __name__ == "__main__":
    main()
