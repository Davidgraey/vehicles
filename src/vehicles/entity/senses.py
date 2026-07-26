import numpy as np
from dataclasses import dataclass
from enum import Enum
from vehicles.entity.angles import calculate_relative_angle, Angle
from vehicles.entity.base_object import BaseObject
import gc

# dataclass constructor for different types?
# enums for type & shape

class SensorType(Enum):
    """ SENSORY MODALITIES """
    SIGHT = 1
    HEARING = 2
    SMELL = 3
    ELECTROMAG = 4
    PRESSURE = 5
    OMNISICENCE = 6


class SensorShape(Enum):
    """ THE SHAPE OF PERCEPTION """
    CONE = "cone"
    CARDIOID = "cardioid"
    BILOBED = "bi-lobed"
    OMNI = "omni"


@dataclass
class Sense:
    type: SensorType
    shape: SensorShape
    range: float | int
    field_of_view: Angle
    noise: float
    xray: float
    falloff_exponent: int = 1.0

    @property
    def is_omni(self) -> bool:
        return self.shape_type == SensorShape.OMNI

    def visualize(self):
        # TODO: add visualize processes in future
        #build up a transparent object based on self.type, self.shape, self.field_of_view, self.rang_max to return.
        pass

    def perceive(self, instance_objects: list) -> np.ndarray:
        """
        Orchestrates the detection and perception of target instances.

        Args:
        ____
        parent_entity: The entity containing this sensor (dependency injection)
        instance_objects: List of GameObjects to detect.

        Returns:
        detected: Array(N) of bools (is the target detected?)
        strength: Array(N) of floats (signal strength 0.0 - 1.0)
        """
        parent_entity = gc.get_referrers(self)
        n_targets = len(instance_objects)
        if n_targets == 0:
            return np.array([]), np.array([])

        # 1. Prepare Vectorized Data
        target_positions = np.array([t.position for t in instance_objects])

        # --- GEOMETRY PHASE ---
        # Calculate relative angles (Position -> Angle)
        # This is the "Geometric Calculation for position"
        relative_angles = calculate_relative_angle(parent_entity.heading, target_positions)

        # Filter by Shape Geometry (Angular Constraints)
        # This determines if the target is "pointing" in the right direction
        geo_mask = evaluate_detection_geometry(
            relative_angles=relative_angles,
            sensor_shape=self.shape,
            sensor_fov=self.field_of_view
        )

        # --- RANGE PHASE ---
        # Calculate Distances from Parent (assuming parent is origin for range calc relative to world)
        # Or calculate distance from Parent to Target
        dists = np.linalg.norm(target_positions - parent_entity.position, axis=1)

        # Filter by Range & Falloff
        # This determines if the target is "close enough" and calculates strength
        range_mask, raw_strength = evaluate_range_falloff(
            dists,
            relative_angles,  # Angle needed for dynamic shapes like Cardioid
            self.shape,
            self.range_sq,
            self.range,
            self.falloff_exponent
        )

        # --- COMBINE PHASE ---
        # A target is detected ONLY if it passes BOTH Geometry and Range checks
        detected = geo_mask & range_mask

        # Clip Strength to valid range
        detected_indices = np.where(detected)[0]
        if len(detected_indices) > 0:
            strength = np.zeros(n_targets)
            strength[detected_indices] = raw_strength[detected_indices]

        return detected, strength

    def __repr__(self):
        return f'{self.type}'

    def __str__(self):
        return f'{self.type}'


def localize_target_coordiantes(entity: GameObject, sensor_heading: np.ndarray, target_pos: np.ndarray) -> np.ndarray:
    """
    Translate target to sensor origin, then rotate into sensor frame.
    Returns array shape (N, 2).
    """
    _position = entity.position
    _dir = entity.direction

    dx = target_pos[:, 0] - _position[0]
    dy = target_pos[:, 1] - _position[1]
    # Rotate -90 deg counter-clockwise to align with heading
    # cos(90)=0, sin(90)=1. Rotating by -(heading - 0)

    cos_h = np.cos(_dir)
    sin_h = np.sin(_dir)

    x_local = dx * cos_h + dy * sin_h
    y_local = -dx * sin_h + dy * cos_h

    return np.column_stack([x_local, y_local])

    def evaluate_detection_geometry(
            relative_angles: np.ndarray,
            sensor_shape: SensorShape,
            sensor_fov: float
    ) -> np.ndarray:
        """
        Checks if targets fall within the angular bounds of the sensor.
        Returns a boolean mask for angular visibility.
        """
        detected_mask = np.zeros(len(relative_angles), dtype=bool)

        if sensor_shape == SensorShape.OMNI:
            # Omni: All directions (360 deg) are technically open, usually handled in Range
            detected_mask = np.ones(len(relative_angles), dtype=bool)

        elif sensor_shape == SensorShape.CONE:
            half_fov = sensor_fov / 2.0
            # Check if angle is within half the FOV of center
            detected_mask = np.abs(relative_angles) <= half_fov

        elif sensor_shape == SensorShape.BILOBED:
            # Front Lobe
            lobe_center = 0.0
            # Back Lobe
            lobe_center = np.pi

            # Normalize angles again specifically for lobes if needed,
            # but abs(angle) covers the center cone logic effectively.
            half_fov = sensor_fov / 2.0

            # Simple check: is it within the cone of center?
            # (Logic expansion required for strict Bi-Lobed cone math in real impl)
            detected_mask = np.abs(relative_angles) <= half_fov

        elif sensor_shape == SensorShape.CARDIOID:
            # Cardioid boundary: dist depends on angle.
            # But for GEOMETRY, we accept that this shape is directionally front-only.
            # We accept angles where cos(theta) is positive (mostly).
            cos_vals = np.cos(relative_angles)
            # Only accept if cos is significantly positive (not behind sensor)
            detected_mask = cos_vals > 0.5  # Threshold for "Front"

        else:
            raise ValueError(f"Unsupported shape: {sensor_shape}")

        return detected_mask


def evaluate_detection_geometry(
            relative_angles: np.ndarray,
            sensor_shape: SensorShape,
            sensor_fov: float
        ) -> np.ndarray:
        """
        Checks if targets fall within the angular bounds of the sensor.
        Returns a boolean mask for angular visibility.
        """
        detected_mask = np.zeros(len(relative_angles), dtype=bool)

        if sensor_shape == SensorShape.OMNI:
            # Omni: All directions (360 deg) are technically open, usually handled in Range
            detected_mask = np.ones(len(relative_angles), dtype=bool)

        elif sensor_shape == SensorShape.CONE:
            half_fov = sensor_fov / 2.0
            # Check if angle is within half the FOV of center
            detected_mask = np.abs(relative_angles) <= half_fov

        elif sensor_shape == SensorShape.BILOBED:
            # Front Lobe
            lobe_center = 0.0
            # Back Lobe
            lobe_center = np.pi

            # Normalize angles again specifically for lobes if needed,
            # but abs(angle) covers the center cone logic effectively.
            half_fov = sensor_fov / 2.0

            # Simple check: is it within the cone of center?
            # (Logic expansion required for strict Bi-Lobed cone math in real impl)
            detected_mask = np.abs(relative_angles) <= half_fov

        elif sensor_shape == SensorShape.CARDIOID:
            # Cardioid boundary: dist depends on angle.
            # But for GEOMETRY, we accept that this shape is directionally front-only.
            # We accept angles where cos(theta) is positive (mostly).
            cos_vals = np.cos(relative_angles)
            # Only accept if cos is significantly positive (not behind sensor)
            detected_mask = cos_vals > 0.5  # Threshold for "Front"

        else:
            raise ValueError(f"Unsupported shape: {sensor_shape}")

        return detected_mask


def evaluate_range_falloff(
            distances: np.ndarray,
            relative_angles: np.ndarray,
            sensor_shape: SensorShape,
            sensor_range_sq: float,
            sensor_range_max: float,
            falloff_exponent: float
    ) -> tuple[np.ndarray, np.ndarray]:
    """
    Checks if targets are within range and calculates signal strength.
    Returns: (bool_mask_for_range, float_array_strength)
    """
    detected_mask = np.zeros(len(distances), dtype=bool)
    strength = np.zeros(len(distances))

    # Determine Falloff Formula based on Shape
    if sensor_shape in (SensorShape.OMNI, SensorShape.CONE):
        # Standard Distance Falloff
        # Check Range
        within_range = (distances ** 2) <= sensor_range_sq  # Using range_sq

        detected_mask = within_range & detected_mask  # Combine with Geometry

        # Calculate Strength
        strength = 1.0 / (distances + 1e-6) ** falloff_exponent
        if sensor_shape == SensorShape.CONE:
            # Cones often have angle-dependent falloff (stronger in center)
            # Add angular falloff modifier to strength calculation if needed
            cos_angles = np.abs(np.cos(relative_angles))
            strength = strength * (cos_angles ** falloff_exponent)

    elif sensor_shape == SensorShape.CARDIOID:
        # Cardioid Falloff is complex (Angle-based radius)
        # Effective Radius = range_max * (1 + cos_theta) / 2

        cos_vals = np.cos(relative_angles)
        factor = (1.0 + cos_vals) / 2.0
        factor = np.clip(factor, 0.0, 1.0)

        # Effective Range check
        effective_range_sq = (sensor_range_max ** 2) * (factor ** 2)

        # Combined check: Angle validity (front) AND Effective Range
        angle_valid = relative_angles < np.pi  # Must be in front to have signal
        detected_mask = (cos_vals > 0) & (distances ** 2 <= effective_range_sq) & angle_valid

        # Strength is purely Distance based for Cardioid (or hybrid, define here)
        # Assuming standard falloff here for strength:
        dist_filtered = distances[detected_mask]
        strength[detected_mask] = 1.0 / (dist_filtered + 1e-6) ** falloff_exponent

    elif sensor_shape == SensorShape.BILOBED:
        # Bi-Lobed implies 2 separate regions, usually symmetric cones
        # Simplified: Logic handled in Geometry or specific Range check

        # Assume Range applies to both lobes
        within_range = (distances ** 2) <= sensor_range_sq
        detected_mask = within_range & detected_mask

        # Strength could be distance + lobe-specific weight (e.g., lobe 1 strength vs lobe 2)
        # For now, treat as standard distance falloff
        strength[detected_mask] = 1.0 / (distances + 1e-6) ** falloff_exponent

    else:
        raise ValueError(f"Unsupported shape: {sensor_shape}")

    return detected_mask, strength


    '''
    what to hold?
    sense type
    sense shape - relative to the entity's facing point (ie vision - in front 45 degrees
    sense range - total distance of effectiveness (include natural falloff over distance)
    sense noise / jitter factor - For example, vision could have less noise but smaller shape / range.  Could also take 
        the form of uncertainty?
    sense x-ray (hearing could avoid obstruction while vision would not.) ( could be a factor? electrosense could be full x-ray; hearing 0.6, vision 0.0
    
    All of these could just be factors / values, so dataclass makes sense.
    '''

