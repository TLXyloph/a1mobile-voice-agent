"""CSV export for any query, and Google Sheets when a credential exists.

The rule this module is built around: **an export never fails because Sheets is
unavailable.** The CSV is the deliverable; Sheets is a convenience on top. So
`export()` writes the file first, then attempts the upload, and reports the
upload's fate as data rather than raising. An owner asking for their numbers at
8:58pm gets a file either way.

Credential discovery, in order:

1. `$GOOGLE_SHEETS_TOKEN` - a raw OAuth access token. The path that works with
   a token minted from the user's already-connected Google account.
2. `sheets.credentials_path` in `config/restaurant.json`, if it holds
   `{"access_token": ...}` or a service-account key **and** google-auth is
   importable.

Neither present is the normal case and is reported as `skipped`, not as an
error - "Sheets was skipped because no credential was found" is a true and
useful sentence; a stack trace is not.
"""

from __future__ import annotations

import csv
import io
import json
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.verticals.restaurant import config as cfg
from src.verticals.restaurant import query as q
from src.verticals.restaurant.store import Store

SHEETS_API = "https://sheets.googleapis.com/v4/spreadsheets"


# -- CSV ------------------------------------------------------------------


def _columns(rows: Sequence[dict[str, Any]]) -> list[str]:
    """Union of keys, first-seen order. Ragged rows must not lose columns."""
    cols: list[str] = []
    for row in rows:
        for key in row:
            if key not in cols:
                cols.append(key)
    return cols


def _cell(value: Any) -> Any:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, default=str)
    return value


def to_csv_string(rows: Sequence[dict[str, Any]]) -> str:
    """Rows -> CSV text. An empty result is a header-less empty string, which
    round-trips back to an empty list rather than to a row of blanks."""
    if not rows:
        return ""
    cols = _columns(rows)
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols, lineterminator="\n", extrasaction="ignore")
    w.writeheader()
    for row in rows:
        w.writerow({c: _cell(row.get(c)) for c in cols})
    return buf.getvalue()


def write_csv(rows: Sequence[dict[str, Any]], path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(to_csv_string(rows), encoding="utf-8")
    return out


def read_csv(path: str | Path) -> list[dict[str, Any]]:
    """Read back what `write_csv` wrote. Used by the round-trip test and by
    anyone who wants to diff two exports."""
    text = Path(path).read_text(encoding="utf-8")
    if not text.strip():
        return []
    return [dict(r) for r in csv.DictReader(io.StringIO(text))]


# -- Google Sheets --------------------------------------------------------


@dataclass
class SheetsResult:
    """What happened with Sheets. Never an exception, always a report."""

    attempted: bool = False
    ok: bool = False
    skipped: bool = True
    reason: str = "not attempted"
    url: str | None = None
    spreadsheet_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempted": self.attempted,
            "ok": self.ok,
            "skipped": self.skipped,
            "reason": self.reason,
            "url": self.url,
            "spreadsheet_id": self.spreadsheet_id,
        }


def find_credential() -> tuple[str | None, str]:
    """Return (access_token, explanation). Token None means "cannot upload"."""
    conf = cfg.default()
    env_name = str(conf.sheets.get("token_env") or "GOOGLE_SHEETS_TOKEN")
    token = os.environ.get(env_name, "").strip()
    if token:
        return token, f"using OAuth token from ${env_name}"

    path = conf.sheets_credentials_path()
    if path is None or not path.exists():
        where = str(path) if path else "config/restaurant.json:sheets.credentials_path"
        return None, f"no credential: ${env_name} unset and {where} not found"

    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"credential file {path.name} unreadable: {exc}"

    if isinstance(data, dict) and data.get("access_token"):
        return str(data["access_token"]), f"using access_token from {path.name}"

    if isinstance(data, dict) and data.get("type") == "service_account":
        try:  # optional dependency, absent in this venv
            from google.auth.transport.requests import Request  # type: ignore
            from google.oauth2 import service_account  # type: ignore
        except ImportError:
            return None, (
                f"{path.name} is a service account key but google-auth is not "
                "installed in this venv; CSV written, Sheets skipped"
            )
        try:
            creds = service_account.Credentials.from_service_account_info(
                data, scopes=["https://www.googleapis.com/auth/spreadsheets"]
            )
            creds.refresh(Request())
            return creds.token, f"service account {data.get('client_email', '?')}"
        except Exception as exc:  # noqa: BLE001 - never take the export down
            return None, f"service account refresh failed: {exc}"

    return None, f"{path.name} holds no usable credential"


def push_to_sheets(
    rows: Sequence[dict[str, Any]],
    *,
    tab: str,
    spreadsheet_id: str | None = None,
    title: str | None = None,
) -> SheetsResult:
    """Write `rows` to a tab. Reports failure; never raises.

    Creates a spreadsheet when no id is configured, so the first export works
    without setup once a token exists.
    """
    conf = cfg.default()
    sheet_id = (spreadsheet_id or str(conf.sheets.get("spreadsheet_id") or "")).strip() or None

    token, why = find_credential()
    if not token:
        return SheetsResult(attempted=False, ok=False, skipped=True, reason=why)

    try:
        import httpx
    except ImportError:  # pragma: no cover - httpx is a hard dep of the app
        return SheetsResult(
            attempted=False, skipped=True, reason="httpx not installed; CSV written only"
        )

    cols = _columns(rows)
    values = [cols] + [[_cell(r.get(c)) for c in cols] for r in rows]
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    try:
        with httpx.Client(timeout=20.0) as client:
            if sheet_id is None:
                created = client.post(
                    SHEETS_API,
                    headers=headers,
                    json={
                        "properties": {
                            "title": title
                            or f"{conf.business_name} - call ledger "
                            f"{datetime.now(timezone.utc):%Y-%m-%d}"
                        },
                        "sheets": [{"properties": {"title": tab}}],
                    },
                )
                if created.status_code >= 400:
                    return SheetsResult(
                        attempted=True,
                        skipped=False,
                        reason=f"create failed {created.status_code}: {created.text[:200]}",
                    )
                sheet_id = created.json().get("spreadsheetId")
            else:
                # Tab may not exist yet. A failure here is fine - the write
                # below will tell us properly.
                client.post(
                    f"{SHEETS_API}/{sheet_id}:batchUpdate",
                    headers=headers,
                    json={"requests": [{"addSheet": {"properties": {"title": tab}}}]},
                )

            client.post(
                f"{SHEETS_API}/{sheet_id}/values/{tab}!A1:clear", headers=headers, json={}
            )
            resp = client.put(
                f"{SHEETS_API}/{sheet_id}/values/{tab}!A1",
                headers=headers,
                params={"valueInputOption": "RAW"},
                json={"values": values},
            )
        if resp.status_code >= 400:
            return SheetsResult(
                attempted=True,
                skipped=False,
                reason=f"write failed {resp.status_code}: {resp.text[:200]}",
                spreadsheet_id=sheet_id,
            )
    except Exception as exc:  # noqa: BLE001 - a network blip is not an export failure
        return SheetsResult(
            attempted=True, skipped=False, reason=f"{type(exc).__name__}: {exc}"
        )

    return SheetsResult(
        attempted=True,
        ok=True,
        skipped=False,
        reason=f"wrote {len(rows)} rows to tab {tab!r} ({why})",
        spreadsheet_id=sheet_id,
        url=f"https://docs.google.com/spreadsheets/d/{sheet_id}",
    )


# -- the one entry point --------------------------------------------------


@dataclass
class ExportResult:
    """The CSV always; the Sheets attempt as a report beside it."""

    name: str
    rows: int
    csv_path: Path
    csv_text: str = ""
    sheets: SheetsResult = field(default_factory=SheetsResult)

    @property
    def summary(self) -> str:
        head = f"{self.rows} row(s) -> {self.csv_path}"
        if self.sheets.ok:
            return f"{head}; Sheets: {self.sheets.url}"
        if self.sheets.skipped:
            return f"{head}; Sheets skipped ({self.sheets.reason})"
        return f"{head}; Sheets FAILED ({self.sheets.reason})"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "rows": self.rows,
            "csv_path": str(self.csv_path),
            "sheets": self.sheets.to_dict(),
            "summary": self.summary,
        }


def export(
    store: Store,
    name: str,
    *,
    directory: str | Path | None = None,
    to_sheets: bool = True,
    spreadsheet_id: str | None = None,
) -> ExportResult:
    """Run a named query, write the CSV, then try Sheets. CSV wins either way."""
    rows = q.run_export(store, name)
    out_dir = Path(directory) if directory else cfg.default().exports_dir
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    path = write_csv(rows, Path(out_dir) / f"{name}-{stamp}.csv")

    sheets = SheetsResult(reason="not requested")
    if to_sheets:
        sheets = push_to_sheets(rows, tab=name, spreadsheet_id=spreadsheet_id)

    return ExportResult(
        name=name,
        rows=len(rows),
        csv_path=path,
        csv_text=to_csv_string(rows),
        sheets=sheets,
    )


def export_all(
    store: Store, *, directory: str | Path | None = None, to_sheets: bool = True
) -> list[ExportResult]:
    return [export(store, name, directory=directory, to_sheets=to_sheets) for name in q.EXPORTS]


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Export restaurant queries to CSV / Sheets")
    ap.add_argument("name", nargs="?", default="all", help=f"one of: all, {', '.join(q.EXPORTS)}")
    ap.add_argument("--db", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-sheets", action="store_true")
    args = ap.parse_args(list(argv) if argv is not None else None)

    st = Store(args.db or cfg.default().database_path)
    names = list(q.EXPORTS) if args.name == "all" else [args.name]
    for n in names:
        print(export(st, n, directory=args.out, to_sheets=not args.no_sheets).summary)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
