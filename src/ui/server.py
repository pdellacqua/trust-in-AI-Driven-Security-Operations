"""
Lightweight Web Server for the Interactive SOC Incident Console.
Built on Python's native http.server with zero external networking dependencies.
"""

import http.server
import socketserver
import os
from pathlib import Path
from typing import Optional


class SOCConsoleRequestHandler(http.server.SimpleHTTPRequestHandler):
    """Serves the static web dashboard files for the SOC console."""

    def __init__(self, *args, **kwargs):
        web_dir = Path(__file__).resolve().parent / "web"
        super().__init__(*args, directory=str(web_dir), **kwargs)

    def end_headers(self):
        # Enable caching controls and CORS
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        super().end_headers()


def start_soc_server(
    host: str = "127.0.0.1",
    port: int = 8080,
    open_browser: bool = False,
) -> None:
    """Starts the interactive SOC triage server."""
    web_dir = Path(__file__).resolve().parent / "web"
    if not web_dir.exists():
        raise FileNotFoundError(f"Web assets directory not found at {web_dir}")

    # Allow port reuse
    socketserver.TCPServer.allow_reuse_address = True

    with socketserver.TCPServer((host, port), SOCConsoleRequestHandler) as httpd:
        url = f"http://{host}:{port}/index.html"
        print(f"\n=======================================================")
        print(f" Trust-Calibrated SOC Incident Console Running")
        print(f" Interactive Dashboard: {url}")
        print(f" Press Ctrl+C to terminate.")
        print(f"=======================================================\n")

        if open_browser:
            import webbrowser
            webbrowser.open(url)

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down SOC console server...")
            httpd.shutdown()


if __name__ == "__main__":
    start_soc_server()
