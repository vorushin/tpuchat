"""CLI client for the Colab code execution server.

Usage:
    uv run python pallas/colab_client.py --health
    uv run python pallas/colab_client.py --code "print(jax.devices())"
    uv run python pallas/colab_client.py --file pallas/my_kernel.py
    uv run python pallas/colab_client.py --file bench.py --timeout 300
    echo "print(cfg)" | uv run python pallas/colab_client.py
"""

import argparse
import json
import sys

import requests


def load_connection(path="pallas/.colab_connection", connect=None):
    """Return (url, token) from connection string or file."""
    if connect:
        raw = connect
    else:
        try:
            raw = open(path).read().strip()
        except FileNotFoundError:
            print(f"Error: connection file not found: {path}", file=sys.stderr)
            print("Save the connection string from Colab server output:", file=sys.stderr)
            print(f'  echo "URL|TOKEN" > {path}', file=sys.stderr)
            sys.exit(1)
    url, token = raw.split("|", 1)
    return url.strip(), token.strip()


def _parse_response(resp):
    """Parse JSON response, show raw body on failure."""
    try:
        return resp.json()
    except Exception:
        print(f"HTTP {resp.status_code} — non-JSON response:", file=sys.stderr)
        print(resp.text[:500], file=sys.stderr)
        sys.exit(1)


def do_health(url, token):
    resp = requests.get(f"{url}/health",
                        headers={"X-Auth-Token": token}, timeout=30)
    data = _parse_response(resp)
    if "error" in data:
        print(f"Error: {data['error']}", file=sys.stderr)
        return 1
    print(f"Status:      {data['status']}")
    print(f"Devices:     {data['devices']}")
    print(f"JAX version: {data['jax_version']}")
    print(f"Config:      {json.dumps(data['config'], indent=2)}")
    return 0


def do_exec(url, token, code, timeout):
    resp = requests.post(f"{url}/exec",
                         headers={"X-Auth-Token": token},
                         json={"code": code},
                         timeout=timeout)
    data = _parse_response(resp)

    if data.get("stdout"):
        print(data["stdout"], end="")

    if data.get("stderr"):
        print("--- stderr ---", file=sys.stderr)
        print(data["stderr"], end="", file=sys.stderr)

    if data.get("error"):
        print("--- ERROR ---", file=sys.stderr)
        print(data["error"], end="", file=sys.stderr)

    if data.get("result") is not None:
        print("--- result ---")
        print(json.dumps(data["result"], indent=2, default=str))

    return 1 if data.get("error") else 0


def main():
    parser = argparse.ArgumentParser(description="Send code to Colab TPU")
    parser.add_argument("--health", action="store_true", help="Health check")
    parser.add_argument("--code", type=str, help="Inline code to execute")
    parser.add_argument("--file", type=str, help="Python file to send")
    parser.add_argument("--timeout", type=int, default=120,
                        help="Request timeout in seconds (default: 120)")
    parser.add_argument("--connect", type=str,
                        help='Connection string "URL|TOKEN" (overrides file)')
    args = parser.parse_args()

    url, token = load_connection(connect=args.connect)

    if args.health:
        sys.exit(do_health(url, token))

    if args.code:
        code = args.code
    elif args.file:
        with open(args.file) as f:
            code = f.read()
    elif not sys.stdin.isatty():
        code = sys.stdin.read()
    else:
        parser.print_help()
        sys.exit(1)

    sys.exit(do_exec(url, token, code, args.timeout))


if __name__ == "__main__":
    main()
