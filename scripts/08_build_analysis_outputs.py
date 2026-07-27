#!/usr/bin/env python3
"""Build complete public-safe aggregate OOF outputs."""

from clinical_domain_mortality.cli import main

raise SystemExit(main(["stage", "--stage", "8", *__import__("sys").argv[1:]]))
