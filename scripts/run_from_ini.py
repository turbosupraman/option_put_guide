#!/usr/bin/env python3
from __future__ import annotations

import configparser
from pathlib import Path
import subprocess
import sys


def parse_bool(value: str, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def resolve_path(base: Path, raw: str, fallback: str) -> Path:
    raw_value = raw.strip() if raw else fallback
    return (base / raw_value).resolve()


def main() -> int:
    ini_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("run.ini")
    if not ini_path.exists():
        print(f"INI file not found: {ini_path}")
        return 1

    config = configparser.ConfigParser()
    config.read(ini_path)

    base_dir = ini_path.resolve().parent
    venv_dir = resolve_path(base_dir, config.get("paths", "venv_dir", fallback=".venv"), ".venv")
    requirements = resolve_path(
        base_dir, config.get("paths", "requirements", fallback="requirements.txt"), "requirements.txt"
    )
    script_path = resolve_path(
        base_dir, config.get("paths", "script", fallback="scripts/filter_sp500.py"), "scripts/filter_sp500.py"
    )

    install_requirements = parse_bool(
        config.get("run", "install_requirements", fallback="true"), True
    )
    run_script = parse_bool(config.get("run", "run_script", fallback="true"), True)

    python_bin = venv_dir / "bin" / "python"
    if not python_bin.exists():
        subprocess.run(["python3", "-m", "venv", str(venv_dir)], check=True)

    if install_requirements and requirements.exists():
        subprocess.run(
            [str(python_bin), "-m", "pip", "install", "-r", str(requirements)],
            check=True,
        )

    if run_script:
        subprocess.run([str(python_bin), str(script_path)], check=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
