"""
Global instances for TORCS Racing AI package.

This module contains global instances to avoid circular imports.
"""

from .ml_models import MLRacingAI
from .visualization import RacingVisualizer

# Global instances
ml_racing_ai = MLRacingAI()
visualizer = RacingVisualizer()