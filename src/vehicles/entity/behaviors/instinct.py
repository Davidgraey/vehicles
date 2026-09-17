"""
Instinct system for behavior-driven agents.

A Behavior pairs a condition with an action:
  - condition(parent, detected_objects) -> List of detected objects that pass the threshold
  - action(parent, triggered_objects)   -> (turn: float, accelerate: float)

Instinct holds a list of Behaviors. Each tick, every Behavior whose condition
returns a non-empty list fires. Their (turn, accelerate) outputs are summed and
clipped to [-1.0, 1.0].

Every Instinct starts empty of custom behaviors, but always has two fallback
behaviors, tried only when no custom behavior fired this tick: "avoid_edges"
(steers back toward the world's center whenever the vehicle is near a
boundary -- needs World.add_entity() to have stamped `world_bounds` onto it)
takes priority, and "wander" (idle drift) is the last resort when nothing
else, including avoid_edges, applies. Add custom reactive behaviors with
.add() -- those always run and sum together, every tick, regardless of the
fallbacks.

Instincts are only evaluated when the vehicle is NOT controlled. The controller
always takes priority — see Vehicle.apply_motor_commands().
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, List, Tuple

from vehicles.entity.angles import calculate_relative_angle, angular_motion_to_cartesian, cartesian_motion_to_angular, Angle, AngularType
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


def always():
    """
    Condition: always fires. Backs the "wander" fallback -- Instinct.evaluate()
    only ever runs wander when nothing else fired this tick, so wander's own
    condition doesn't need to inspect detected_objects itself (checking for
    "nothing detected" there was the bug: senses can report low-strength/
    irrelevant detections that no other behavior's condition accepts, which
    left both wander AND everything else silent -- vehicle stalls dead).
    """
    def condition(parent, detected_objects: list) -> list:
        return [True]
    return condition


def same_type(entity_type: type, min_strength: float = 0.0, max_distance: float = None):
    """
    Condition: keep only detections whose live object is an instance of
    `entity_type`. This is the instance-type check group behaviors need --
    it's what lets, say, a Fish's flock/separation react only to other
    Fish, ignoring predators, scenery, or any other Vehicle subclass
    sharing the world. Detections with no live 'object' reference (plain
    sensor output with nothing to type-check) are dropped.

    `max_distance`, if given, additionally drops anything farther than
    that from the parent -- handy for carving a tight "personal space"
    radius for separation out of the same (wider-range) sense a looser
    flock/cohesion behavior also reads from, without needing a second Sense.
    """
    def condition(parent, detected_objects: list) -> list:
        result = [
            d for d in detected_objects
            if isinstance(d.get('object'), entity_type) and d['strength'] > min_strength
        ]
        if max_distance is not None:
            result = [
                d for d in result
                if np.linalg.norm(np.array(d['position']) - parent.position) <= max_distance
            ]
        return result
    return condition


def not_type(entity_type: type, min_strength: float = 0.0, max_distance: float = None):
    """
    Condition: keep only detections whose live object is NOT an instance of
    `entity_type` -- the complement of same_type(). This is what lets a
    Predator pursue anything that isn't another Predator without having to
    enumerate every prey species by name; a new species added later is
    automatically huntable just by not being a Predator.

    Same optional `min_strength`/`max_distance` filters as same_type().
    Detections with no live 'object' reference are dropped (nothing to
    type-check).
    """
    def condition(parent, detected_objects: list) -> list:
        result = [
            d for d in detected_objects
            if d.get('object') is not None
            and not isinstance(d['object'], entity_type)
            and d['strength'] > min_strength
        ]
        if max_distance is not None:
            result = [
                d for d in result
                if np.linalg.norm(np.array(d['position']) - parent.position) <= max_distance
            ]
        return result
    return condition


def hungry_for(entity_type: type, hunger_threshold: float = 20.0, min_strength: float = 0.0, max_distance: float = None):
    """
    Condition: same_type(entity_type), but only once the parent is hungry
    enough (parent.hunger >= hunger_threshold). Backs "find food" -- it
    steps aside once hunger's below threshold, the same way wander steps
    aside once anything else is detected.
    """
    def condition(parent, detected_objects: list) -> list:
        if getattr(parent, 'hunger', 0.0) < hunger_threshold:
            return []
        return same_type(entity_type, min_strength=min_strength, max_distance=max_distance)(parent, detected_objects)
    return condition


def near_edge(margin: float = 60.0):
    """
    Condition: fires when the parent is within `margin` units of any world
    boundary. Needs `parent.world_bounds` (stamped on by World.add_entity())
    to know how big the world is -- silently never fires without it, same
    as hungry_for() silently never fires without `parent.hunger`.
    """
    def condition(parent, detected_objects: list) -> list:
        bounds = getattr(parent, 'world_bounds', None)
        if bounds is None:
            return []
        width, height = bounds
        x, y = parent.position
        near = x < margin or x > width - margin or y < margin or y > height - margin
        return [True] if near else []
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
    """
    Turn toward the direction AWAY from the strongest detected object, and
    flee forward.

    (Turning away and reversing cancels out into net movement TOWARD the
    threat, so instead we compute the away-direction as its own angle and
    move forward into it, mirroring how separation() does it.)
    """
    threat = max(triggered, key=lambda d: d['strength'])
    threat_pos = np.array(threat['position'])

    away_delta = parent.position - threat_pos
    distance = np.linalg.norm(away_delta)
    if distance > 1e-6:
        away_angle = Angle(AngularType.RADIANS, float(np.arctan2(away_delta[0], away_delta[1])))
    else:
        away_angle = parent.heading  # threat is exactly on top of us -- no direction to flee, hold heading

    parent.heading.cast_as_radians()
    relative = calculate_relative_angle(parent.heading, away_angle)
    turn = np.clip(relative.value, -0.3, 0.3)

    return turn, 1.0


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


def arrive(parent, triggered: list, slow_radius: float = 100.0) -> Tuple[float, float]:
    """
    Nature of Code's "arrive": seek the strongest detection like `approach`
    does, but taper the throttle down as the distance shrinks instead of
    always flooring it. Desired speed ramps from 0 up to full linearly
    across `slow_radius`, so the vehicle settles near the target instead of
    overshooting -- `accelerate(0)` already decays velocity on its own (see
    BaseObject.accelerate), so tapering toward 0 is enough to coast to a
    stop; no separate braking/reverse term needed.
    """
    target = max(triggered, key=lambda d: d['strength'])
    target_pos = np.array(target['position'])

    target_angle = get_facing_angle(parent.position, target_pos)
    parent.heading.cast_as_radians()
    relative = calculate_relative_angle(parent.heading, target_angle)
    turn = np.clip(relative.value, -0.3, 0.3)

    distance = np.linalg.norm(target_pos - parent.position)
    accel = float(np.clip(distance / slow_radius, 0.0, 1.0))

    return turn, accel


def pursuit(parent, triggered: list, lookahead: float = 1.0) -> Tuple[float, float]:
    """
    Nature of Code's "pursuit": seek where the strongest detection is
    HEADING, not where it currently is. Extrapolates the target's position
    forward by its own current heading/speed (same conversion `move()`
    uses internally), then steers toward that predicted point exactly like
    `approach` steers toward a static one.

    Falls back to the target's current position if `triggered` didn't
    carry a live object reference (older/plain detections), or if the
    target isn't moving -- there's nothing to predict.
    """
    target = max(triggered, key=lambda d: d['strength'])
    target_obj = target.get('object')

    target_pos = np.array(target['position'])
    if target_obj is not None and getattr(target_obj, 'velocity', 0.0):
        target_obj.heading.cast_as_radians()
        predicted_delta = angular_motion_to_cartesian(target_obj.heading, target_obj.velocity * lookahead)
        target_pos = target_pos + predicted_delta

    target_angle = get_facing_angle(parent.position, target_pos)
    parent.heading.cast_as_radians()
    relative = calculate_relative_angle(parent.heading, target_angle)
    turn = np.clip(relative.value, -0.3, 0.3)

    return turn, 1.0


def separation(parent, triggered: list, strength: float = 0.5) -> Tuple[float, float]:
    """
    Nature of Code's "separation": unlike every other action here, which
    reacts to just the strongest single detection, separation reacts to
    ALL of `triggered` at once -- pair it with a condition like
    gate_strength() that returns the whole qualifying set, not closest(1).

    For each neighbor, builds a vector pointing away from it and weights
    it by 1/distance (closer neighbors push harder), then averages all of
    those into one combined push-away direction and turns toward it.
    """
    push = np.zeros(2)
    count = 0

    for d in triggered:
        away = parent.position - np.array(d['position'])
        distance = np.linalg.norm(away)
        if distance > 1e-6:
            push += (away / distance) / distance  # unit vector, weighted by 1/distance
            count += 1

    if count == 0:
        return 0.0, 0.0

    push /= count
    push_angle = Angle(AngularType.RADIANS, float(np.arctan2(push[0], push[1])))

    parent.heading.cast_as_radians()
    relative = calculate_relative_angle(parent.heading, push_angle)
    turn = np.clip(relative.value, -0.3, 0.3)

    return turn, strength


def flock(parent, triggered: list, align_weight: float = 1.0, cohesion_weight: float = 1.0, speed: float = 0.5) -> Tuple[float, float]:
    """
    Nature of Code's remaining two boid rules, combined: alignment (match
    the group's average heading) and cohesion (steer toward the group's
    average position) -- together with separation(), this is full
    Reynolds flocking. Like separation, reacts to the whole `triggered`
    list at once, not just the strongest detection;
    pairs its condition with same_type() so a vehicle only flocks with its own kind.

    (cohesion: direction to the neighbors' centroid; alignment: avg of neighbors heading vectors)
    """
    positions = np.array([d['position'] for d in triggered])
    centroid = positions.mean(axis=0)

    cohesion_delta = centroid - parent.position
    cohesion_distance = np.linalg.norm(cohesion_delta)
    cohesion_vector = cohesion_delta / cohesion_distance if cohesion_distance > 1e-6 else np.zeros(2)

    heading_vectors = []
    for d in triggered:
        neighbor = d.get('object')
        if neighbor is not None:
            neighbor.heading.cast_as_radians()
            heading_vectors.append(angular_motion_to_cartesian(neighbor.heading, 1.0))

    if heading_vectors:
        align_vector = np.mean(heading_vectors, axis=0)
        align_norm = np.linalg.norm(align_vector)
        if align_norm > 1e-6:
            align_vector = align_vector / align_norm
    else:
        align_vector = np.zeros(2)

    combined = cohesion_vector * cohesion_weight + align_vector * align_weight
    combined_norm = np.linalg.norm(combined)

    if combined_norm < 1e-6:
        # surrounded evenly / nothing to align to -- hold heading, just cruise
        return 0.0, speed

    target_angle, _ = cartesian_motion_to_angular(tuple(combined))

    parent.heading.cast_as_radians()
    relative = calculate_relative_angle(parent.heading, target_angle)
    turn = np.clip(relative.value, -0.3, 0.3)

    return float(turn), speed


def wander(parent, triggered: list, turn_strength: float = 0.2, jitter: float = 0.05, accel: float = 0.3) -> Tuple[float, float]:
    """
    Default fallback: amble forward on a lazily drifting heading so an idle
    vehicle reads as alive rather than frozen or locked into a fixed circle
    """
    bias = getattr(parent, '_wander_bias', 0.0)
    bias += np.random.uniform(-jitter, jitter)
    bias = float(np.clip(bias, -turn_strength, turn_strength))
    parent._wander_bias = bias

    return bias, accel+bias


def avoid_edges(parent, triggered: list) -> Tuple[float, float]:
    """
    Turn toward the world's center and push forward. Pairs with
    near_edge() -- steers a vehicle back inland before it ever reaches
    World.apply_bounds()'s hard position clip, which otherwise just pins
    it at the boundary tick after tick with nothing to turn it back.
    """
    width, height = parent.world_bounds
    center = np.array([width / 2.0, height / 2.0])

    center_angle = get_facing_angle(parent.position, center)
    parent.heading.cast_as_radians()
    relative = calculate_relative_angle(parent.heading, center_angle)
    turn = np.clip(relative.value, -0.3, 0.3)

    return turn, 1.0


# ====================================
# BEHAVIOR
# pairs a condition with an action. Every named behavior (wander included)
# ends up as one of these -- see Instinct.add().

@dataclass
class Behavior:
    name: str
    condition: Callable
    action: Callable
    enabled: bool = True

    def decide(self, parent, detected_objects: list) -> Tuple[float, float] | None:
        """
        run condition. If it returns objects, run action and return commands
        None if not met
        """
        if not self.enabled:
            return None
        triggered = self.condition(parent, detected_objects)
        if not triggered:
            return None
        return self.action(parent, triggered)


def _default_fallbacks() -> List[Behavior]:
    """
    Tried, in order, only when no custom behavior fired this tick:
    "avoid_edges" first (steers back toward the world's center whenever the
    vehicle is near a boundary), then "wander" (idle drift) as the last
    resort once even avoid_edges doesn't apply.
    """
    return [
        Behavior(name="avoid_edges", condition=near_edge(), action=avoid_edges),
        Behavior(name="wander", condition=always(), action=wander),
    ]


# ========================================================
# INSTINCT - a collection of behaviors

@dataclass
class Instinct:
    behaviors: List[Behavior] = field(default_factory=list)
    fallbacks: List[Behavior] = field(default_factory=_default_fallbacks)
    active_behaviors: List[str] = field(default_factory=list)

    def add(self, name: str, condition: Callable, action: Callable) -> 'Instinct':
        """Add a custom behavior -- always evaluated, sums with the rest."""
        self.behaviors.append(Behavior(name=name, condition=condition, action=action))
        return self

    def enable(self, name: str) -> None:
        for b in self.behaviors + self.fallbacks:
            if b.name == name:
                b.enabled = True

    def disable(self, name: str) -> None:
        for b in self.behaviors + self.fallbacks:
            if b.name == name:
                b.enabled = False

    def evaluate(self, parent, detected_objects: list) -> Tuple[float, float]:
        """
        Sum the outputs of every custom behavior whose condition fires.
        If none fire, use the first fallback that fires instead (avoid_edges,
        then wander) -- not just when detected_objects is empty, since
        senses can report detections too weak/irrelevant for any custom
        behavior's own condition to accept. Without this fallback a vehicle
        in that state would get no turn/accelerate from anything and stall.
        Names of the behaviors that fired this tick are stashed onto
        `active_behaviors`, for anyone (e.g. the renderer) who wants to show
        what a vehicle is currently doing.
        Returns (turn, speeed) clipped to [-1.0, 1.0].
        """
        turn_sum = 0.0
        accel_sum = 0.0
        self.active_behaviors = []

        for behavior in self.behaviors:
            result = behavior.decide(parent, detected_objects)
            if result is not None:
                turn_sum += result[0]
                accel_sum += result[1]
                self.active_behaviors.append(behavior.name)

        if not self.active_behaviors:
            for fallback in self.fallbacks:
                result = fallback.decide(parent, detected_objects)
                if result is not None:
                    turn_sum += result[0]
                    accel_sum += result[1]
                    self.active_behaviors.append(fallback.name)
                    break

        return (
            float(np.clip(turn_sum, -1.0, 1.0)),
            float(np.clip(accel_sum, -1.0, 1.0))
        )
