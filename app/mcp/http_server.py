from typing import Any, Optional
import httpx
from fastmcp import FastMCP

mcp     = FastMCP("http-tools")

_base_url: str = ""
_headers : dict = {}


def configure(base_url: str, headers: Optional[dict] = None):
    global _base_url, _headers
    _base_url = base_url.rstrip("/")
    _headers  = headers or {}


@mcp.tool
async def http_get(path: str, params: Optional[dict] = None) -> dict:
    """Make a GET request to the backend API."""
    async with httpx.AsyncClient(headers=_headers, timeout=30) as client:
        resp = await client.get(f"{_base_url}{path}", params=params)
        return {
            "status_code"   : resp.status_code,
            "body"          : resp.json() if resp.headers.get("content-type", "").startswith("application/json") else resp.text,
        }


@mcp.tool
async def http_post(path: str, body: Optional[dict] = None) -> dict:
    """Make a POST request to the backend API."""
    async with httpx.AsyncClient(headers=_headers, timeout=30) as client:
        resp = await client.post(f"{_base_url}{path}", json=body)
        return {
            "status_code": resp.status_code,
            "body"       : resp.json() if resp.headers.get("content-type", "").startswith("application/json") else resp.text,
        }


@mcp.tool
async def http_put(path: str, body: Optional[dict] = None) -> dict:
    """Make a PUT request to the backend API."""
    async with httpx.AsyncClient(headers=_headers, timeout=30) as client:
        resp = await client.put(f"{_base_url}{path}", json=body)
        return {
            "status_code": resp.status_code,
            "body"       : resp.json() if resp.headers.get("content-type", "").startswith("application/json") else resp.text,
        }


@mcp.tool
async def http_delete(path: str, params: Optional[dict] = None) -> dict:
    """Make a DELETE request to the backend API."""
    async with httpx.AsyncClient(headers=_headers, timeout=30) as client:
        resp = await client.delete(f"{_base_url}{path}", params=params)
        return {
            "status_code": resp.status_code,
            "body"       : resp.json() if resp.headers.get("content-type", "").startswith("application/json") else resp.text,
        }

if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8002)