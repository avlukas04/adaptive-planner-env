"""
Google Calendar integration for LifeOps.

Fetches events from the user's primary Google Calendar for display in the week view.
Requires:
  - pip install google-api-python-client google-auth-oauthlib
  - OAuth credentials from Google Cloud Console (Calendar API enabled)
  - Credentials JSON path in GOOGLE_CREDENTIALS_PATH or default ./credentials.json

Usage:
  from app.gcal_client import get_week_events
  events = get_week_events()  # Returns list of dicts with start_min, end_min, title, location
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _get_credentials_path() -> Optional[Path]:
    path = os.environ.get("GOOGLE_CREDENTIALS_PATH")
    if path:
        return Path(path)
    return _REPO_ROOT / "credentials.json"


def get_week_events(
    week_start: Optional[datetime] = None,
    days: int = 7,
) -> List[Dict[str, Any]]:
    """
    Fetch events from Google Calendar for the given week.

    Returns list of dicts: {event_id, title, start_min, end_min, location, kind}
    start_min/end_min are minutes since midnight of the event's day (for single-day view compatibility).
    """
    try:
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from google.auth.transport.requests import Request
        from googleapiclient.discovery import build
    except ImportError:
        return []

    creds_path = _get_credentials_path()
    if not creds_path or not creds_path.is_file():
        return []

    SCOPES = ["https://www.googleapis.com/auth/calendar.readonly"]
    token_path = _REPO_ROOT / "token.json"

    creds = None
    if token_path.is_file():
        creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(str(creds_path), SCOPES)
            creds = flow.run_local_server(port=0)
        if token_path:
            token_path.parent.mkdir(parents=True, exist_ok=True)
            with open(token_path, "w") as f:
                f.write(creds.to_json())

    service = build("calendar", "v3", credentials=creds)
    if week_start is None:
        week_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        week_start -= timedelta(days=week_start.weekday())
    time_min = week_start.isoformat() + "Z"
    time_max = (week_start + timedelta(days=days)).isoformat() + "Z"

    events_result = (
        service.events()
        .list(
            calendarId="primary",
            timeMin=time_min,
            timeMax=time_max,
            singleEvents=True,
            orderBy="startTime",
        )
        .execute()
    )
    events_raw = events_result.get("items", [])

    out: List[Dict[str, Any]] = []
    for e in events_raw:
        start = e.get("start", {})
        end = e.get("end", {})
        start_dt = start.get("dateTime") or start.get("date")
        end_dt = end.get("dateTime") or end.get("date")
        if not start_dt or not end_dt:
            continue
        if "T" in start_dt:
            dt_s = datetime.fromisoformat(start_dt.replace("Z", "+00:00"))
            dt_e = datetime.fromisoformat(end_dt.replace("Z", "+00:00"))
        else:
            dt_s = datetime.fromisoformat(start_dt)
            dt_e = datetime.fromisoformat(end_dt)
        # Convert to minutes since midnight of event day
        day_start = dt_s.replace(hour=0, minute=0, second=0, microsecond=0)
        start_min = int((dt_s - day_start).total_seconds() / 60)
        end_min = int((dt_e - day_start).total_seconds() / 60)
        out.append({
            "event_id": e.get("id", ""),
            "title": e.get("summary", "Untitled"),
            "start_min": start_min,
            "end_min": end_min,
            "location": e.get("location", ""),
            "kind": "meeting",
        })
    return out


def is_gcal_available() -> bool:
    """True if GCal credentials are configured and google packages are installed."""
    creds_path = _get_credentials_path()
    if not creds_path or not creds_path.is_file():
        return False
    try:
        from google.oauth2.credentials import Credentials  # noqa: F401
        from google_auth_oauthlib.flow import InstalledAppFlow  # noqa: F401
        return True
    except ImportError:
        return False
