"""Tiny helper: verify graphiti_core resolves to vendor/graphiti, exit 1 otherwise."""
import sys

import graphiti_core

if "vendor" not in graphiti_core.__file__.lower():
    print(f"[ERROR] graphiti_core resolves to: {graphiti_core.__file__}")
    print("        Expected somewhere under vendor/graphiti/")
    sys.exit(1)
print(f"[OK] graphiti_core: {graphiti_core.__file__}")
