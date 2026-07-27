#!/usr/bin/env bash
set -euo pipefail
clinical-domain-mortality validate --config configs/mimic.example.yaml
clinical-domain-mortality run --dataset mimic --config configs/mimic.example.yaml
