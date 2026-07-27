#!/usr/bin/env python3
"""Run through frozen cohort, outcome, and baseline construction."""

from clinical_domain_mortality.cli import main

raise SystemExit(main(["stage", "--stage", "2", *__import__("sys").argv[1:]]))
