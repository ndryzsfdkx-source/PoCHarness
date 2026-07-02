"""B finalization gate for the A+B+C synthesis scaffold."""
from .config import FinalizationReviewConfig
from .reviewer import FinalizationReviewer, create_finalization_reviewer

__all__ = ["FinalizationReviewConfig", "FinalizationReviewer", "create_finalization_reviewer"]
