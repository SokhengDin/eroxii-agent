from datetime import datetime
from typing import Optional

import httpx
from fastmcp import FastMCP

from app.core.config import settings
from app.core.logger import logger

mcp = FastMCP("alpr-tools")


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(base_url=settings.API_BASE_URL, timeout=15)


def _fmt_dt(raw: str) -> str:
    if not raw:
        return "—"
    try:
        return datetime.fromisoformat(raw).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return raw


def _fmt_user(user: dict) -> str:
    added_by = user.get("subscription_added_by") or {}
    return (
        f"Name        : {user.get('first_name', '')} {user.get('last_name', '')} (@{user.get('username', '')})\n"
        f"Type        : {user.get('user_type', '')}  |  Active: {'Yes' if user.get('is_active') else 'No'}\n"
        f"Phone       : {user.get('phone_number', '—')}\n"
        f"Payment Due : {'Yes' if user.get('payment_needed') else 'No'}\n"
        f"Vehicle     : {user.get('vehicle_type', '')} — {user.get('license_plate', '')} ({user.get('plate_type', '')})\n"
        f"Last Entry  : {_fmt_dt(user.get('last_entry', ''))}\n"
        f"Last Exit   : {_fmt_dt(user.get('last_exit', ''))}\n"
        f"Subscription: {'Active' if user.get('has_subscription') else 'None'}\n"
        f"Sub Start   : {_fmt_dt(user.get('subscription_started_at', ''))}\n"
        f"Sub Expires : {_fmt_dt(user.get('subscription_expired_at', ''))}\n"
        f"Added By    : {added_by.get('full_name', '—')} (@{added_by.get('username', '—')})\n"
        f"Member Since: {_fmt_dt(user.get('created_at', ''))}"
    )


def _fmt_session(rec: dict) -> str:
    plate   = rec.get("license_plate", "")
    user    = rec.get("user", {}) or {}
    vehicle = rec.get("vehicle", {}) or {}
    entry   = rec.get("latest_entry_capture", {}) or {}
    exit_   = rec.get("latest_exit_capture", {}) or {}
    name    = f"{user.get('first_name', '')} {user.get('last_name', '')}".strip()
    model   = f"{vehicle.get('model', '')} {vehicle.get('series', '')}".strip()

    lines = [
        f"Plate       : {plate}",
        f"Name        : {name}",
        f"Type        : {user.get('user_type', '')}",
        f"Vehicle     : {vehicle.get('vehicle_type', '')} — {model}",
        f"Status      : {rec.get('current_status', '')}",
    ]
    if entry.get("captured_at"):
        lines.append(f"Entry       : {_fmt_dt(entry['captured_at'])}  confidence={entry.get('confidence_score', '')}")
    if exit_.get("captured_at"):
        lines.append(f"Exit        : {_fmt_dt(exit_['captured_at'])}  confidence={exit_.get('confidence_score', '')}")
    return "\n".join(lines)


# ── Resources — parameterized reads, agent reads these by URI ─────────────────

@mcp.resource("alpr://member/{license_plate}")
async def resource_member(license_plate: str) -> str:
    """Registered member info for a license plate."""
    logger.info(f"[ALPR resource] member: {license_plate}")
    async with _client() as http:
        resp = await http.get(
            "/api/v1/users/members/search",
            params={"search_term": license_plate, "user_type": "ALL", "skip": 0, "limit": 1},
        )
    if resp.status_code != 200:
        return f"Backend error {resp.status_code}"

    users = resp.json().get("users", [])
    if not users:
        return f"Plate {license_plate} not registered."

    lines = [f"Plate: {license_plate} — {len(users)} result(s)\n"]
    for u in users:
        lines.append(_fmt_user(u))
        lines.append("")
    return "\n".join(lines).strip()


@mcp.resource("alpr://plate/{license_plate}/full-info")
async def resource_plate_full_info(license_plate: str) -> str:
    """Full info for one exact plate — user, vehicle, latest entry/exit captures."""
    logger.info(f"[ALPR resource] plate full-info: {license_plate}")
    async with _client() as http:
        resp = await http.get(f"/api/v1/alpr-tracking/plate-full-info/{license_plate}")
    if resp.status_code != 200:
        return f"Backend error {resp.status_code}"
    return _fmt_session(resp.json())


@mcp.resource("alpr://plate/{license_plate}/latest-detection")
async def resource_latest_detection(license_plate: str) -> str:
    """Latest live detection for a plate — is_valid flag, user, vehicle, captures."""
    logger.info(f"[ALPR resource] latest-detection: {license_plate}")
    async with _client() as http:
        resp = await http.get(f"/api/v1/alpr-tracking/latest-detection/{license_plate}")
    if resp.status_code != 200:
        return f"Backend error {resp.status_code}"
    data = resp.json()
    return f"Valid: {'Yes' if data.get('is_valid') else 'No'}\n{_fmt_session(data)}"


@mcp.resource("alpr://plate/{license_plate}/captures")
async def resource_captures(license_plate: str) -> str:
    """Entry and exit capture images for a plate (latest 10)."""
    logger.info(f"[ALPR resource] captures: {license_plate}")
    async with _client() as http:
        resp = await http.get(
            f"/api/v1/alpr-tracking/captures/{license_plate}",
            params={"skip": 0, "limit": 10},
        )
    if resp.status_code != 200:
        return f"Backend error {resp.status_code}"

    data  = resp.json()
    lines = [f"Captures for {license_plate}:"]
    for c in data if isinstance(data, list) else [data]:
        lines.append(
            f"  [{c.get('capture_type', '')}] {c.get('captured_at', '')}  "
            f"confidence={c.get('confidence_score', '')}"
        )
    return "\n".join(lines)


@mcp.resource("alpr://plate/{license_plate}/session-history")
async def resource_session_history(license_plate: str) -> str:
    """Parking session history for a plate (latest 10 sessions)."""
    logger.info(f"[ALPR resource] session-history: {license_plate}")
    async with _client() as http:
        resp = await http.get(
            f"/api/v1/alpr-tracking/plate-detailed-records/{license_plate}",
            params={"skip": 0, "limit": 10},
        )
    if resp.status_code != 200:
        return f"Backend error {resp.status_code}"

    records = resp.json()
    if not records:
        return f"No session records for {license_plate}."

    lines = [f"Session history for {license_plate}:\n"]
    for r in records if isinstance(records, list) else [records]:
        lines.append(
            f"  Entry: {r.get('entry_time', '')}  Exit: {r.get('exit_time', '')}  "
            f"Duration: {r.get('duration', '')}  Payment: {r.get('payment_status', '')}"
        )
    return "\n".join(lines)


@mcp.resource("alpr://latest-records")
async def resource_latest_records() -> str:
    """Most recent parking records across all plates (latest 10)."""
    logger.info("[ALPR resource] latest-records")
    async with _client() as http:
        resp = await http.get("/api/v1/alpr-tracking/latest-records/", params={"limit": 10})
    if resp.status_code != 200:
        return f"Backend error {resp.status_code}"

    records = resp.json()
    if not records:
        return "No recent records."

    lines = [f"Latest {len(records)} record(s):\n"]
    for r in records if isinstance(records, list) else [records]:
        lines.append(_fmt_session(r))
        lines.append("")
    return "\n".join(lines).strip()


# ── Resource bridge — lets agent read any resource by URI ─────────────────────

@mcp.tool
async def read_resource(uri: str) -> str:
    """Read any ALPR resource by URI. Available URIs:
    - alpr://member/{license_plate}
    - alpr://plate/{license_plate}/full-info
    - alpr://plate/{license_plate}/latest-detection
    - alpr://plate/{license_plate}/captures
    - alpr://plate/{license_plate}/session-history
    - alpr://latest-records
    """
    logger.info(f"[ALPR tool] read_resource: {uri}")

    import re as _re

    if m := _re.fullmatch(r'alpr://member/(.+)', uri):
        return await resource_member(m.group(1))
    if m := _re.fullmatch(r'alpr://plate/(.+)/full-info', uri):
        return await resource_plate_full_info(m.group(1))
    if m := _re.fullmatch(r'alpr://plate/(.+)/latest-detection', uri):
        return await resource_latest_detection(m.group(1))
    if m := _re.fullmatch(r'alpr://plate/(.+)/captures', uri):
        return await resource_captures(m.group(1))
    if m := _re.fullmatch(r'alpr://plate/(.+)/session-history', uri):
        return await resource_session_history(m.group(1))
    if uri == 'alpr://latest-records':
        return await resource_latest_records()

    return f"Unknown resource URI: {uri}"


# ── Tools — search actions where agent constructs the query ───────────────────

@mcp.tool
async def search_member_by_plate(license_plate: str) -> str:
    """Look up a registered member by license plate. Returns member info and subscription status."""
    logger.info(f"[ALPR tool] search_member_by_plate: {license_plate}")
    async with _client() as http:
        resp = await http.get(
            "/api/v1/users/members/search",
            params={"search_term": license_plate, "user_type": "ALL", "skip": 0, "limit": 1},
        )
    if resp.status_code != 200:
        return f"Backend error {resp.status_code}"

    users = resp.json().get("users", [])
    if not users:
        return f"Plate {license_plate} not registered in the system."

    lines = [f"Plate: {license_plate} — {len(users)} result(s)\n"]
    for u in users:
        lines.append(_fmt_user(u))
        lines.append("")
    return "\n".join(lines).strip()


@mcp.tool
async def search_plate_full_info(search_term: str, limit: int = 5) -> str:
    """Search by plate number OR user name. Returns full session info with latest captures. Use when unsure if input is a name or plate."""
    logger.info(f"[ALPR tool] search_plate_full_info: {search_term}")
    async with _client() as http:
        resp = await http.get(
            f"/api/v1/alpr-tracking/search-with-full-info/{search_term}",
            params={"limit": limit},
        )
    if resp.status_code != 200:
        return f"Backend error {resp.status_code}"

    records = resp.json()
    if not records:
        return f"No records found for: {search_term}"

    lines = [f"Results for '{search_term}' — {len(records)} found\n"]
    for r in records if isinstance(records, list) else [records]:
        lines.append(_fmt_session(r))
        lines.append("")
    return "\n".join(lines).strip()


@mcp.tool
async def search_parking_records(
    search_term: str,
    skip: int = 0,
    limit: int = 10,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> str:
    """Free-text search across plate, name, or vehicle model. Use start_date/end_date (ISO 8601) to filter by date range."""
    logger.info(f"[ALPR tool] search_parking_records: {search_term} {start_date}→{end_date}")
    params: dict = {"search_term": search_term, "skip": skip, "limit": limit}
    if start_date:
        params["start_date"] = start_date
    if end_date:
        params["end_date"] = end_date

    async with _client() as http:
        resp = await http.get("/api/v1/license-plates/search/", params=params)
    if resp.status_code != 200:
        return f"Backend error {resp.status_code}"

    records = resp.json()
    if not records:
        return f"No parking records found for: {search_term}"

    lines = [f"Parking records for '{search_term}' — {len(records)} result(s):\n"]
    for r in records if isinstance(records, list) else [records]:
        lines.append(
            f"  Plate: {r.get('license_plate', '')}  "
            f"Entry: {r.get('entry_time', '')}  Exit: {r.get('exit_time', '')}  "
            f"Payment: {r.get('payment_status', '')}"
        )
    return "\n".join(lines)


@mcp.tool
async def get_latest_records(limit: int = 10, vehicle_type: Optional[str] = None) -> str:
    """Get recent parking records across all plates. vehicle_type: CAR, MOTORCYCLE, TRUCK, BUS, VAN."""
    logger.info(f"[ALPR tool] get_latest_records vehicle_type={vehicle_type}")
    params: dict = {"skip": 0, "limit": limit}
    if vehicle_type:
        params["vehicle_type"] = vehicle_type

    async with _client() as http:
        resp = await http.get("/api/v1/alpr-tracking/latest-records/", params=params)
    if resp.status_code != 200:
        return f"Backend error {resp.status_code}"

    records = resp.json()
    if not records:
        return "No recent records."

    lines = [f"Latest {len(records)} record(s):\n"]
    for r in records if isinstance(records, list) else [records]:
        lines.append(_fmt_session(r))
        lines.append("")
    return "\n".join(lines).strip()


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8003)
