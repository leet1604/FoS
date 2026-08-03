"""Convenience alias for the current A/B -> C handoff.

The Stage B result already contains the Stage A context ID and audit trail, so
Stage C can be rerun without repeating Stage A/B. This script delegates to
``run_stage_c.py`` and exists to make the full-pipeline handoff explicit.
"""

from run_stage_c import main


if __name__ == "__main__":
    main()
