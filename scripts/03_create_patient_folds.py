#!/usr/bin/env python3
"""Run through deterministic five-fold patient assignment."""

from clinical_domain_mortality.cli import main

raise SystemExit(main(["stage", "--stage", "3", *__import__("sys").argv[1:]]))
