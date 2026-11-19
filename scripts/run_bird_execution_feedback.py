"""Run BIRD strategy C with at most one SQL-error-driven revision."""

import sys

from run_bird_tool_assisted import main


if __name__ == "__main__":
    sys.exit(main(repair_on_sql_error=True))
