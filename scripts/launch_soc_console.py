#!/usr/bin/env python3
"""
Launcher for the Interactive Trust-Calibrated SOC Incident Console.
Usage:
    python3 scripts/launch_soc_console.py --port 8080
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ui.server import start_soc_server


def parse_args():
    parser = argparse.ArgumentParser(description="Launch Interactive SOC Analyst Console")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface")
    parser.add_argument("--port", type=int, default=8080, help="Port to serve console")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open browser")
    return parser.parse_args()


def main():
    args = parse_args()
    start_soc_server(
        host=args.host,
        port=args.port,
        open_browser=not args.no_browser,
    )


if __name__ == "__main__":
    main()
