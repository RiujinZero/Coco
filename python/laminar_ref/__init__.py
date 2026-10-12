"""Laminar 200k organism — NumPy reference implementation (the slower faithful kernel)."""
from .config import Params
from .organism import Organism
from .tasks import SyntheticVocab

__all__ = ["Params", "Organism", "SyntheticVocab"]
