from datetime import datetime

import httpx
from fastmcp import FastMCP

from app.core.config import settings
from app.core.logger import logger

mcp = FastMCP("alpr-tools")


def _format_user(user: dict) -> str:
    expired_raw = user.get("subscription_expired_at", "")
    try:
        expired = datetime.fromisoformat(expired_raw).strftime("%Y-%m-%d %H:%M")
    except Exception:
        expired = expired_raw

    return (
        f"Name        : {user.get('first_name', '')} {user.get('last_name', '')}\n"
        f"Type        : {user.get('user_type', '')}\n"
        f"Phone       : {user.get('phone_number', '')}\n"
        f"Payment Due : {'Yes' if user.get('payment_needed') else 'No'}\n"
        f"Vehicle     : {user.get('vehicle_type', '')} — {user.get('license_plate', '')}\n"
        f"Plate Type  : {user.get('plate_type', '')}\n"
        f"Subscription: {'Active' if user.get('has_subscription') else 'None'}\n"
        f"Expires     : {expired}"
    )


@mcp.tool
async def search_member_by_plate(license_plate: str) -> str:
    """Search for a registered member by vehicle license plate number. Returns member info or not-found message."""
    logger.info(f"[ALPR MCP] Searching plate: {license_plate}")
    async with httpx.AsyncClient(timeout=15) as http:
        resp = await http.get(
            f"{settings.API_BASE_URL}/api/v1/users/members/search",
            params={
                "search_term": license_plate,
                "user_type"  : "ALL",
                "skip"       : 0,
                "limit"      : 1,
            },
        )

    logger.info(f"[ALPR MCP] Backend response: {resp.status_code}")
    if resp.status_code != 200:
        return f"Backend error {resp.status_code} for plate: {license_plate}"

    users = resp.json().get("users", [])
    logger.info(f"[ALPR MCP] Found {len(users)} user(s) for plate: {license_plate}")
    if not users:
        return f"Plate {license_plate} not registered in the system."

    lines = [f"Plate: {license_plate} — {len(users)} result(s) found\n"]
    for u in users:
        lines.append(_format_user(u))
        lines.append("")
    return "\n".join(lines).strip()


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8003)
