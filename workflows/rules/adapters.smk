rule run_adapter:
    output:
        "outputs/synthetic_run/adapter/.done"
    shell:
        "mkdir -p outputs/synthetic_run/adapter && touch {output}"
