"""
TORCS AI Racing Package

A sophisticated machine learning-based racing AI for TORCS (The Open Racing Car Simulator).
Features advanced ML models, automated training, real-time visualization, and continuous learning.
"""

__version__ = "2.0.0"
__author__ = "GitHub Copilot Enhanced"
__description__ = "Advanced ML Racing AI for TORCS"

from .client import Client
from .ml_models import MLRacingAI
from .training import (
    automated_training_pipeline,
    continuous_learning_mode,
    perfection_training_pipeline,
    elite_training_curriculum,
    intensive_training_session
)
from .visualization import RacingVisualizer
from .utils import start_torcs_server, analyze_ml_models, generate_racing_insights
from .globals import ml_racing_ai, visualizer

__all__ = [
    'Client',
    'MLRacingAI',
    'RacingVisualizer',
    'automated_training_pipeline',
    'continuous_learning_mode',
    'perfection_training_pipeline',
    'elite_training_curriculum',
    'intensive_training_session',
    'start_torcs_server',
    'analyze_ml_models',
    'generate_racing_insights'
]