"""Local recording inspection. No viewer server or external service."""

import argparse
import json
from pathlib import Path

from .exporters import PerfettoExporter
from .identifiers import TraceId
from .models import TraceQuery
from .stores.sqlite import SQLiteTraceStore


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("list", "summary", "export", "recover"):
        sub = commands.add_parser(name)
        sub.add_argument("database", type=Path)
        if name in ("summary", "export"):
            sub.add_argument("trace_id")
        if name == "export":
            sub.add_argument("destination", type=Path)
        if name == "list":
            sub.add_argument("--limit", type=int, default=100)
            sub.add_argument("--offset", type=int, default=0)
    evals_cmd = commands.add_parser("evals", help="local evaluation artifacts and gates")
    evals_cmd.add_argument("args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command == "evals":
        from .evals.cli import main as evals_main

        return evals_main(args.args)
    if not args.database.is_file():
        parser.error("recording database does not exist")
    store = SQLiteTraceStore(args.database, read_only=args.command != "recover")
    try:
        if args.command == "list":
            print(
                json.dumps(
                    [
                        s.to_dict()
                        for s in store.list_traces(TraceQuery(limit=args.limit, offset=args.offset))
                    ]
                )
            )
        elif args.command == "recover":
            print(json.dumps({"recovered_spans": store.recover_unfinished()}))
        else:
            bundle = store.get_trace(TraceId.from_hex(args.trace_id))
            if bundle is None:
                parser.error("trace not found")
            if args.command == "export":
                PerfettoExporter().export([bundle], args.destination, diagnostics=store.diagnostics)
            else:
                print(
                    json.dumps(
                        {
                            "trace_id": str(bundle.trace.trace_id),
                            "name": bundle.trace.name,
                            "spans": len(bundle.spans),
                            "events": sum(len(s.events) for s in bundle.spans),
                            "metrics": len(bundle.metrics),
                            "diagnostics": store.diagnostics,
                        }
                    )
                )
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
