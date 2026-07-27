#!/usr/bin/env bash
set -euo pipefail
python3 synthetic_data/generate_synthetic_data.py
python3 -m clinical_domain_mortality synthetic-run
python3 -m clinical_domain_mortality verify --run-dir outputs/synthetic
