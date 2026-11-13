"""Explicit developer/CI provisioning, never imported or invoked by tests.

Pinned to the linux-amd64 manifest in google/perfetto tools/trace_processor v59.1.
"""

import argparse
import hashlib
import os
import tempfile
import urllib.request
from pathlib import Path

VERSION = "v59.1"
URL = (
    f"https://commondatastorage.googleapis.com/perfetto-luci-artifacts/{VERSION}/"
    "linux-amd64/trace_processor_shell"
)
SHA256 = "2434129a3af4dc9e2034a476f7b76447b490975213e301769310d3bd0d5d6458"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    destination = parser.parse_args().destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(URL, timeout=60) as response:
        payload = response.read()
    if hashlib.sha256(payload).hexdigest() != SHA256:
        raise RuntimeError("Perfetto download checksum mismatch")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
        temporary.chmod(0o755)
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(f"Provisioned Perfetto {VERSION}: {destination}")


if __name__ == "__main__":
    main()
