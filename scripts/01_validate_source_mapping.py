#!/usr/bin/env python3
"""Run source configuration and standardized-schema validation."""

from clinical_domain_mortality.cli import main

raise SystemExit(main(["stage", "--stage", "1", *__import__("sys").argv[1:]]))
