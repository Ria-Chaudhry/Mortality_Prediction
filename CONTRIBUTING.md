# Contributing

Thank you for contributing. This repository is designed for reproducible clinical-domain comparisons across sites, so please keep changes modular and auditable.

## Development Workflow

1. Create a feature branch.
2. Add or update tests for core behavior.
3. Keep dataset-specific SQL, identifiers, and code systems inside `adapters/` or `configs/`.
4. Run `pytest` before opening a pull request.
5. Do not commit restricted patient-level data.

## Adapter Contributions

Adapters must produce the standardized schemas in `schemas/`. New adapters should include:

- an example config with no credentials;
- mapping templates or documented mapping requirements;
- synthetic or de-identified fixtures;
- integration tests that exercise the adapter contract.
