"""Load KEY=VALUE lines from the project .env into os.environ (existing vars win)."""
import os
from pathlib import Path

# ponytail: minimal .env loader; python-dotenv if quoting/multiline ever needed
for line in (Path(__file__).parent.parent / ".env").read_text().splitlines():
    if "=" in line and not line.lstrip().startswith("#"):
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())
