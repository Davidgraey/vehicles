"""
Instinct system for behavior-driven agents.

A Behavior pairs a condition with an action:
  - condition(parent, detected_objects) -> List of detected objects that pass the threshold
  - action(parent, triggered_objects)   -> (turn: float, accelerate: float)

Instinct holds a list of Behaviors. Each tick, every Behavior whose condition
returns a non-empty list fires. Their (turn, accelerate) outputs are summed and
clipped to [-1.0, 1.0].

Every Instinct starts with one default Behavior, "wander": the opposite of every
other behavior here, its condition fires only when NOTHING is detected, so it's
what a vehicle falls back to when it has no stimulus to react to. Add more
reactive behaviors on top with .add(); wander steps aside automatically the
moment anything else has something to detect.

Instincts are only evaluated when the vehicle is NOT controlled. The controller
always takes priority — see Vehicle.apply_motor_commands().
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, List, Tuple

from vehicles.entity.angles import calculate_relative_angle, Angle, AngularType
from vehicles.entity.base_object import get_facing_angle


# ==========================================
# CONDITIONAL PROCESSES
# Each returns the subset of detected_objects that pass

def gate_strength(threshold: float = 0.01):
    """Condition: detected objects whose signal strength exceeds thresh """
    def condition(parent, detected_objects: list) -> list:
        return [d for d in detected_objects if d['strength'] > threshold]
    return condition


def closest(n: int = 1):
    """Condition: the n strongest detections above threshold """
    def condition(parent, detected_objects: list) -> list:
        return sorted(detected_objects, key=lambda d: d['strength'], reverse=True)[:n]
    return condition


def nothing_detected():
    """
    Condition: fires only when the vehicle's senses detected nothing at all
    this tick. This is what backs the default "wander" behavior -- it's the
    inverse of every other condition here, which fire based on what WAS
    detected.
    """
    def condition(parent, detected_objects: list) -> list:
        return [] if detected_objects else [True]
    return condition


# =====================================
# ACTION FUNCTIONS

def approach(parent, triggered: list) -> Tuple[float, float]:
    """Turn toward and accel at the strongest detected object."""
    target = max(triggered, key=lambda d: d['strength'])
    target_pos = np.array(target['position'])

    target_angle = get_facing_angle(parent.position, target_pos)
    parent.heading.cast_as_radians()
    relative = calculate_relative_angle(parent.heading, target_angle)

    turn = np.clip(relative.value, -0.3, 0.3)
    return turn, 1.0


def evade(parent, triggered: list) -> Tuple[float, float]:
    """Turn away from and reverse from the strongest detected object."""
    threat = max(triggered, key=lambda d: d['strength'])
    threat_pos = np.array(threat['position'])

    threat_angle = get_facing_angle(parent.position, threat_pos)
    parent.heading.cast_as_radians()
    relative = calculate_relative_angle(parent.heading, threat_angle)

    turn = np.clip(-relative.value, -0.3, 0.3)
    return turn, -1.0


def orbit(parent, triggered: list, radius: float = 100.0) -> Tuple[float, float]:
    """Circle around the strongest detected object at a fixed distance."""
    target = max(triggered, key=lambda d: d['strength'])
    target_pos = np.array(target['position'])
    distance = np.linalg.norm(target_pos - parent.position)

    target_angle = get_facing_angle(parent.position, target_pos)
    orbit_target = target_angle + Angle(AngularType.RADIANS, np.pi / 2)

    parent.heading.cast_as_radians()
    relative = calculate_relative_angle(parent.heading, orbit_target)
    turn = np.clip(relative.value, -0.3, 0.3)

    if distance < radius * 0.8:
        accel = -0.5
    elif distance > radius * 1.2:
        accel = 0.5
    else:
        accel = 0.2

    return turn, accel


def wander(parent, triggered: list, turn_strength: float = 0.2, jitter: float = 0.05, accel: float = 0.3) -> Tuple[float, float]:
    """
    Default fallback: amble forward on a lazily drifting heading so an idle
    vehicle reads as alive rather than frozen or locked into a fixed circle.

    Nudges a persistent per-vehicle turn bias by a small random amount each
    tick and clips it to +/-turn_strength, so the heading wanders smoothly
    instead of jittering frame to frame. The bias lives on `parent` itself
    (Vehicle instances are deep-copied on World.reset(), so it resets cleanly
    along with everything else).
    """
    bias = getattr(parent, '_wander_bias', 0.0)
    bias += np.random.uniform(-jitter, jitter)
    bias = float(np.clip(bias, -turn_strength, turn_strength))
    parent._wander_bias = bias

    return bias, accel


# ====================================
# BEHAVIOR
# pairs a condition with an action.
# we'll likely have to rework these as we go.

@dataclass
class Behavior:
    name: str
    condition: Callable
    action: Callable
    enabled: bool = True

    def decide(self, parent, detected_objects: list) -> Tuple[float, float] | None:
        """
        run condition. If it returns objects, run action and return commands
        None if condition not met
        """
        if not self.enabled:
            return None
        triggered = self.condition(parent, detected_objects)
        if not triggered:
            return None
        return self.action(parent, triggered)


def _default_behaviors() -> List[Behavior]:
    """
    Every Instinct starts with just "wander" enabled -- the idle fallback.
    It only ever fires when detected_objects is empty, so as soon as a
    vehicle has something real to react to, wander steps aside on its own.
    """
    return [Behavior(name="wander", condition=nothing_detected(), action=wander)]


# ========================================================
# INSTINCT - a collection of behaviors

@dataclass
class Instinct:
    behaviors: List[Behavior] = field(default_factory=_default_behaviors)

    def add(self, name: str, condition: Callable, action: Callable) -> 'Instinct':
        """Add a behavior"""
        self.behaviors.append(Behavior(name=name, condition=condition, action=action))
        return self

    def enable(self, name: str) -> None:
        for b in self.behaviors:
            if b.name == name:
                b.enabled = True

    def disable(self, name: str) -> None:
        for b in self.behaviors:
            if b.name == name:
                b.enabled = False

    def evaluate(self, parent, detected_objects: list) -> Tuple[float, float]:
        """
        Evaluate all behaviors. Sum outputs of every behavior whose condition fires.
        (With no detections, only "wander" -- the default -- will fire; every
        other behavior's own condition naturally returns nothing to act on.)
        Returns (turn, speeed) clipped to [-1.0, 1.0].
        """
        turn_sum = 0.0
        accel_sum = 0.0

        for behavior in self.behaviors:
            result = behavior.decide(parent, detected_objects)
            if result is not None:
                turn_sum += result[0]
                accel_sum += result[1]

        return (
            float(np.clip(turn_sum, -1.0, 1.0)),
            float(np.clip(accel_sum, -1.0, 1.0))
        )
