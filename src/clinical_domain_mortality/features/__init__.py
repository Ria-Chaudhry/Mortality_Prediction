"""First-24-hour event linkage and fold-specific feature construction."""

from .construction import DomainFeatures, build_fold_domain_features
from .linkage import PreparedEvents, prepare_domain_events
from .matrices import assemble_matrix
from .selection import ConceptSelection, select_concepts
from .validation import assert_no_forbidden_features

__all__ = [
    "ConceptSelection",
    "DomainFeatures",
    "PreparedEvents",
    "assemble_matrix",
    "assert_no_forbidden_features",
    "build_fold_domain_features",
    "prepare_domain_events",
    "select_concepts",
]
