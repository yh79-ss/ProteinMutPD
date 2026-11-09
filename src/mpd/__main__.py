"""`python -m mpd` is the same as the `mpd` command."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
