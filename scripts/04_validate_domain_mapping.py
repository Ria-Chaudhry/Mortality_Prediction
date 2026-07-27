#!/usr/bin/env python3
"""Run through event mapping and semantics validation."""

from clinical_domain_mortality.cli import main

raise SystemExit(main(["stage", "--stage", "4", *__import__("sys").argv[1:]]))
