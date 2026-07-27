#!/usr/bin/env python3
"""Run through first-24-hour event preparation without global selection."""

from clinical_domain_mortality.cli import main

raise SystemExit(main(["stage", "--stage", "5", *__import__("sys").argv[1:]]))
