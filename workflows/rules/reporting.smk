rule synthetic_reproduction:
    output:
        "outputs/synthetic_run/metrics.csv"
    shell:
        "python scripts/reproduce_manuscript.py --config examples/synthetic/config.yaml --output-dir outputs/synthetic_run"
