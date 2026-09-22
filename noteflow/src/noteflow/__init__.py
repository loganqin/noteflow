"""Public API for the NoteFlow inference framework."""

from .core import (
    AssignmentResult,
    NoteFlowConfig,
    NoteFlowResult,
    RankingResult,
    fixed_first_rankings,
    max_weight_assignment,
    run_noteflow,
)
from .soft_capacity import SoftCapacityResult, fit_soft_capacity

__all__ = [
    "AssignmentResult",
    "NoteFlowConfig",
    "NoteFlowResult",
    "RankingResult",
    "SoftCapacityResult",
    "fit_soft_capacity",
    "fixed_first_rankings",
    "max_weight_assignment",
    "run_noteflow",
]

__version__ = "0.1.0"

