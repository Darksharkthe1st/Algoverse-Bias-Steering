"""Enable `python -m src.bias_steer ...` (delegates to the CLI, §7.3)."""

from .cli import main

raise SystemExit(main())
