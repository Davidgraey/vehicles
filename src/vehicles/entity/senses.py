import numpy as np
from dataclasses import dataclass
from enum import Enum
from vehicles.entity.angles import Angle

CARDIOID_THRESHOLD = 0.5


class SensorType(Enum):
    SIGHT = 1
    HEARING = 2
    SMELL = 3
    ELECTROMAG = 4
    PRESSURE = 5
    OMNISICENCE = 6


class SensorShape(Enum):
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
    noise: float = 0.0
    xray: float = 0.0
    falloff_exponent: int = 2.0

    @property
    def is_omni(self) -> bool:
        return self.shape == SensorShape.OMNI

    def detect(self, distances: np.ndarray, relative_angles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Parameters
        ----------
        distances : (M,) distance from the sensing vehicle to each candidate
        relative_angles : (M,) signed angle from the vehicle's heading to each candidate, radians

        Returns
        -------
        detected : (M,) bool
        strength : (M,) float, zero where not detected
        """
        geo_mask = evaluate_detection_geometry(
            relative_angles=relative_angles,
            sensor_shape=self.shape,
            sensor_fov=self.field_of_view.value
        )
        range_mask, raw_strength = evaluate_range_falloff(
            distances=distances,
            relative_angles=relative_angles,
            sensor_shape=self.shape,
            sensor_range_max=self.range,
            falloff_exponent=self.falloff_exponent
        )

        detected = geo_mask & range_mask
        strength = np.where(detected, raw_strength, 0.0)

        if self.noise > 0:
            strength[detected] += np.random.uniform(-self.noise, self.noise, size=int(detected.sum()))

        return detected, strength

    def angular_extent(self) -> list[tuple[float, float]]:
        """
        Visible angular intervals, relative to the heading.

        Returns
        -------
        list of (start, end) in radians, ordered by start, each with start < end
        """
        half = self.field_of_view.to_radians() / 2.0

        if self.shape == SensorShape.OMNI:
            return [(-np.pi, np.pi)]
        if self.shape == SensorShape.CONE:
            return [(-half, half)]
        if self.shape == SensorShape.CARDIOID:
            limit = float(np.arccos(CARDIOID_THRESHOLD))
            return [(-limit, limit)]
        if self.shape == SensorShape.BILOBED:
            return [(-half, half), (np.pi - half, np.pi + half)]

        raise ValueError(f"Unsupported shape: {self.shape}")

    def __repr__(self):
        return f'{self.type.value}: {self.shape.value}'

    def __str__(self):
        return f'{self.type}'


def evaluate_detection_geometry(
        relative_angles: np.ndarray,
        sensor_shape: SensorShape,
        sensor_fov: float
) -> np.ndarray:
    """
    Checks if targets fall within the angular bounds of the sensor.
    sensor_fov is the FULL field of view (e.g., π/2 = 90 degrees)
    relative_angles is angle from parent_heading TO target (0 = same direction as heading)

    The sensor cone is centered on the parent's heading.
    """
    n = len(relative_angles)
    detected_mask = np.zeros(n, dtype=bool)
    half_fov = sensor_fov / 2.0

    if sensor_shape == SensorShape.OMNI:
        # Omni: 360 degrees, all angles detected
        detected_mask = np.ones(n, dtype=bool)

    elif sensor_shape == SensorShape.CONE:
        # Cone is centered on parent heading
        # detected if relative angle is within [-FOV/2, +FOV/2] of center
        detected_mask = (relative_angles >= -half_fov) & (relative_angles <= half_fov)

    elif sensor_shape == SensorShape.CARDIOID:
        # Cardioid: only front direction (cos(θ) > 0.5)
        cos_vals = np.cos(relative_angles)
        detected_mask = cos_vals > CARDIOID_THRESHOLD

    elif sensor_shape == SensorShape.BILOBED:
        # Two lobes: front cone around 0, back cone around π
        front_detected = np.abs(relative_angles) <= half_fov
        back_detected = np.abs(relative_angles - np.pi) <= half_fov
        detected_mask = front_detected | back_detected

    else:
        raise ValueError(f"Unsupported shape: {sensor_shape}")

    return detected_mask


def evaluate_range_falloff(
        distances: np.ndarray,
        relative_angles: np.ndarray,
        sensor_shape: SensorShape,
        sensor_range_max: float,
        falloff_exponent: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate whether targets are within range and compute signal strength.
    Returns (detected_mask, strength_array).
    """
    n = len(distances)

    if sensor_shape == SensorShape.OMNI:
        # Simple distance check
        detected_mask = (distances <= sensor_range_max)

        # Strength = 1 / distance^exponent
        strength = np.ones(n) / (distances + 1e-6) ** falloff_exponent
        strength = np.clip(strength, 0.0, 1.0)

    elif sensor_shape == SensorShape.CONE:
        # For cones, range is uniform in the cone, 0 outside
        within_range = (distances <= sensor_range_max)
        detected_mask = within_range  # Geometry already filtered angles

        # Strength decays with distance, uniform in angle
        strength = np.ones(n) / (distances + 1e-6) ** falloff_exponent
        strength = np.clip(strength, 0.0, 1.0)

    elif sensor_shape == SensorShape.CARDIOID:
        within_range = (distances <= sensor_range_max)
        detected_mask = within_range  # Geometry already filtered angles

        # Strength decays with distance, uniform in angle
        strength = np.ones(n) / (distances + 1e-6) ** falloff_exponent
        strength = np.clip(strength, 0.0, 1.0)

    elif sensor_shape == SensorShape.BILOBED:
        within_range = (distances <= sensor_range_max)
        detected_mask = within_range  # Geometry already filtered angles

        # Strength decays with distance, uniform in angle
        strength = np.ones(n) / (distances + 1e-6) ** falloff_exponent
        strength = np.clip(strength, 0.0, 1.0)

    else:
        raise ValueError(f"Unsupported shape: {sensor_shape}")

    return detected_mask, strength
