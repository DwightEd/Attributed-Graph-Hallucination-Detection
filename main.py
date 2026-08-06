"""Repository entry point.

New subcommands are provided by :mod:`topology_flow.cli`.  Unknown legacy
arguments are delegated to the former ``attention_graph.cli`` during the
migration period.
"""

from topology_flow.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
