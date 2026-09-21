"""
Brain: a predictive-model counterpart to Behavior.

A Behavior pairs a condition with an action -- a rule. A Brain pairs a name
with a model instead: no condition, the model itself decides what to output
from the observation. Same call shape as Behavior (name, enabled, decide()
returning (turn, accelerate) or None), so a Brain drops into
Instinct.behaviors or Instinct.fallbacks right alongside rule-based
Behaviors -- Instinct.evaluate() only ever calls .decide(), it doesn't care
which kind produced it.
"""
from dataclasses import dataclass
from typing import Callable, Optional, Tuple


@dataclass
class Brain:
    """
    Parameters
    ----------
    name : identifying label, same role as Behavior.name
    model : callable, (parent, detected_objects) -> (turn, accelerate)
    enabled : same role as Behavior.enabled
    """
    name: str
    model: Callable[[object, list], Tuple[float, float]]
    enabled: bool = True

    def decide(self, parent, detected_objects: list) -> Optional[Tuple[float, float]]:
        """Run the model. Returns None when disabled."""
        if not self.enabled:
            return None
        return self.model(parent, detected_objects)
