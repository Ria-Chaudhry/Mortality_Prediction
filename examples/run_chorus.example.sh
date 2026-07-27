#!/usr/bin/env bash
set -euo pipefail
clinical-domain-mortality validate --config configs/chorus.example.yaml
clinical-domain-mortality run --dataset chorus --config configs/chorus.example.yaml
