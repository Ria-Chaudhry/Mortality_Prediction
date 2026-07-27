#!/usr/bin/env python3
"""Run all 160 outer-fold fits and validate OOF coverage."""

from clinical_domain_mortality.cli import main

raise SystemExit(main(["stage", "--stage", "7", *__import__("sys").argv[1:]]))
