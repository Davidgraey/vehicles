"""
A single tick to capture the world's sensory inputs
I called this Perception last time - that was confusing me with Observation (what an entity has perceived) and Senses
    (the parameters of the sense itself
The pairwise distances and relative angles are computed once per tick and each entity's sense reads its own row of
them.
Everything measured (states, geometry and what was detected) is captured as a Frame of arrays indexed by entity.

Observations as per-vehicle egocentirc view are a condensation of the Frame view
"""
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from vehicles.entity.angles import relative_geometry
from vehicles.entity.base_object import BaseObject
from vehicles.entity.observation import Detection, Observation
from vehicles.entity.vehicle import Vehicle
from vehicles.world.terrain import Terrain


@dataclass(frozen=True)
class Frame:
    """
    Everything measured in one tick, indexed by entity (N) and terrain sample (K).

    Parameters
    ----------
    tick : tick at which the world was sensed
    ids : (N,) entity ids
    positions : (N, 2)
    headings : (N,) radians
    velocities, health, hunger : (N,) entity state; 0 where an entity has none
    previous_actions : (N, 2) (turn, accelerate) applied in the previous tick; 0 where an entity has none
    distances : (N, N) distance from i to j
    relative_angles : (N, N) signed angle from i's heading to j, radians
    detected : (N, N) True where i's sense detected j
    strength : (N, N) detected signal strength, 0 where not detected
    terrain_samples : (K, 2) boundary points of all terrain, concatenated
    terrain_distances, terrain_relative_angles, terrain_detected, terrain_strength : (N, K)
        the same four measurements from each entity to each terrain boundary sample
    """
    tick: int
    ids: np.ndarray
    positions: np.ndarray
    headings: np.ndarray
    velocities: np.ndarray
    health: np.ndarray
    hunger: np.ndarray
    previous_actions: np.ndarray
    distances: np.ndarray
    relative_angles: np.ndarray
    detected: np.ndarray
    strength: np.ndarray
    terrain_samples: np.ndarray
    terrain_distances: np.ndarray
    terrain_relative_angles: np.ndarray
    terrain_detected: np.ndarray
    terrain_strength: np.ndarray


def sense_world(entities: List[BaseObject], terrain: List[Terrain], tick: int) -> Frame:
    """
    Measure and sense the whole world once.

    Parameters
    ----------
    entities : every entity in the world
    terrain : every terrain region in the world
    tick : current tick

    Returns
    -------
    Frame
    """
    count = len(entities)
    positions = np.array([e.position for e in entities], dtype=np.float64).reshape(-1, 2)
    headings = np.array([e.heading.to_radians() for e in entities], dtype=np.float64)
    samples = np.vstack([t.samples for t in terrain]) if terrain else np.empty((0, 2))

    distances, relative_angles = relative_geometry(positions, headings, np.vstack([positions, samples]))
    terrain_distances, terrain_relative_angles = distances[:, count:], relative_angles[:, count:]
    distances, relative_angles = distances[:, :count], relative_angles[:, :count]

    detected = np.zeros((count, count), dtype=bool)
    strength = np.zeros((count, count))
    terrain_detected = np.zeros(terrain_distances.shape, dtype=bool)
    terrain_strength = np.zeros(terrain_distances.shape)

    # TODO: validate the assumption -> idxs are consistent and statically defined.
    indices = np.arange(count)

    for i, entity in enumerate(entities):
        if not isinstance(entity, Vehicle):
            continue

        others = indices[indices != i]
        detected[i, others], strength[i, others] = entity.sense.detect(
            distances[i, others],
            relative_angles[i, others],
        )
        terrain_detected[i], terrain_strength[i] = entity.sense.detect(
            terrain_distances[i],
            terrain_relative_angles[i],
        )

    return Frame(
        tick=tick,
        ids=np.array([e.id for e in entities], dtype=np.int64),
        positions=positions,
        headings=headings,
        velocities=np.array([e.velocity for e in entities], dtype=np.float64),
        health=np.array([e.health for e in entities], dtype=np.float64),
        hunger=np.array([getattr(e, 'hunger', 0.0) for e in entities], dtype=np.float64),
        previous_actions=np.array([getattr(e, 'last_action', (0.0, 0.0)) for e in entities], dtype=np.float64).reshape(-1, 2),
        distances=distances,
        relative_angles=relative_angles,
        detected=detected,
        strength=strength,
        terrain_samples=samples,
        terrain_distances=terrain_distances,
        terrain_relative_angles=terrain_relative_angles,
        terrain_detected=terrain_detected,
        terrain_strength=terrain_strength,
    )


def build_observations(entities: List[BaseObject], terrain: List[Terrain], frame: Frame) -> List[
    Tuple[Vehicle, Observation]]:
    """
    Reduce a Frame to each vehicle's Observation.

    Parameters
    ----------
    entities : every entity in the world, in the order the Frame was sensed
    terrain : every terrain region in the world, in the order the Frame was sensed
    frame : result of sense_world()

    Returns
    -------
    list of (vehicle, observation)
    """
    bounds = np.concatenate([[0], np.cumsum([len(t.samples) for t in terrain])]).astype(int)

    observations = []
    for i, entity in enumerate(entities):
        if not isinstance(entity, Vehicle):
            continue

        detections = tuple(
            Detection(
                position=(float(frame.positions[j, 0]), float(frame.positions[j, 1])),
                strength=float(frame.strength[i, j]),
                kind=type(entities[j]),
            )
            for j in np.flatnonzero(frame.detected[i])
        )

        terrain_detections = []
        for region, start, end in zip(terrain, bounds[:-1], bounds[1:]):
            sensed = frame.terrain_detected[i, start:end]
            if not sensed.any():
                continue
            nearest = start + np.argmin(np.where(sensed, frame.terrain_distances[i, start:end], np.inf))
            terrain_detections.append(Detection(
                position=(float(frame.terrain_samples[nearest, 0]), float(frame.terrain_samples[nearest, 1])),
                strength=float(frame.terrain_strength[i, nearest]),
                kind=type(region),
            ))

        observations.append((entity, Observation(detections=detections, terrain=tuple(terrain_detections))))

    return observations
