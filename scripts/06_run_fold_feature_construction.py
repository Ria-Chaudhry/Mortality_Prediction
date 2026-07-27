#!/usr/bin/env python3
"""Run through training-fold-specific feature construction."""

from clinical_domain_mortality.cli import main

raise SystemExit(main(["stage", "--stage", "6", *__import__("sys").argv[1:]]))
