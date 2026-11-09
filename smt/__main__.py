"""Command line:

    python -m smt refresh [--months 36] [--no-holdings]  download + validate + store
    python -m smt check                                  data-quality report (exit 1 on failure)
    python -m smt report [--out FINDINGS.md]             write the findings, computed from the data
    python -m smt serve [--port 8050]                    run the dashboard
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

from smt import checks, pipeline, settings, store


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="smt", description="Smart Money Tracker")
    ap.add_argument("--db", default=None, help=f"SQLite path (default {settings.DB_PATH})")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("refresh", help="download, validate and store all data")
    r.add_argument("--months", type=int, default=settings.HOLDINGS_MONTHS, help="month-end portfolios to keep")
    r.add_argument("--no-holdings", action="store_true", help="skip portfolio disclosures")
    sub.add_parser("check", help="run data-quality checks")
    rp = sub.add_parser("report", help="write the findings, computed from the data")
    rp.add_argument("--out", default=str(settings.ROOT / "FINDINGS.md"))
    s = sub.add_parser("serve", help="run the dashboard")
    s.add_argument("--port", type=int, default=int(os.getenv("PORT", "8050")))
    s.add_argument("--host", default=os.getenv("HOST", "127.0.0.1"))
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.cmd == "refresh":
        rep = pipeline.refresh(args.db, months=args.months, holdings=not args.no_holdings)
        print(f"verified funds : {rep.verified}")
        print(f"NAV rows       : {rep.nav_rows:,}")
        print(f"files parsed   : {rep.files}")
        print(f"holding rows   : {rep.holdings_rows:,}")
        for m in rep.mismatched:
            print(f"MISMATCH  {m}")
        for e in rep.errors:
            print(f"ERROR     {e}")
        return 1 if rep.mismatched else 0

    if args.cmd == "check":
        con = store.connect(args.db)
        df = checks.run(con)
        con.close()
        for row in df.itertuples():
            print(f"[{row.status.upper():4}] {row.check}: {row.detail}")
        return 1 if (df["status"] == "fail").any() else 0

    if args.cmd == "report":
        from smt import report
        from smt.dashboard.data import Data
        data = Data.load(args.db)
        if data.empty:
            print("No data yet. Run:  python -m smt refresh", file=sys.stderr)
            return 1
        print(f"Wrote {report.write(data, args.out)}")
        return 0

    from smt.dashboard.app import create_app  # imported late: Dash is heavy
    create_app(args.db, warm=True).run(host=args.host, port=args.port, debug=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
