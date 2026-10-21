"""
The last T encoded ticks for one species, batched as (B, T, D).
"""
from typing import Tuple

import numpy as np

from vehicles.learning.encoding import EncodingSpec


class History:
    """
    Fixed-size ring of encoded rows.

    Parameters
    ----------
    spec : EncodingSpec
    batch_size : B, vehicles of the species
    """
    def __init__(self, spec: EncodingSpec, batch_size: int):
        self.spec = spec
        self.batch_size = batch_size
        self.selves = np.zeros((spec.history, batch_size, spec.self_dim), dtype=np.float32)
        self.grids = np.zeros((spec.history, batch_size, spec.cells), dtype=np.int8)
        self.count = 0

    def record(self, self_block: np.ndarray, grid: np.ndarray) -> None:
        """Store one tick from ObservationEncoder.encode."""
        slot = self.count % self.spec.history
        self.selves[slot] = self_block
        self.grids[slot] = grid
        self.count += 1

    def clear(self) -> None:
        """Forget every recorded tick."""
        self.selves[:] = 0
        self.grids[:] = 0
        self.count = 0

    def batch(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns
        -------
        x : (B, T, D) float32, oldest tick first, left-padded with zeros
        valid : (B, T) bool, False on padding
        """
        history, start = self.spec.history, self.spec.grid_start
        filled = min(self.count, history)

        x = np.zeros((self.batch_size, history, self.spec.dim), dtype=np.float32)
        valid = np.zeros((self.batch_size, history), dtype=bool)
        if filled == 0:
            return x, valid

        slots = np.arange(self.count - filled, self.count) % history
        x[:, history - filled:, :start] = self.selves[slots].transpose(1, 0, 2)
        x[:, history - filled:, start:] = self.grids[slots].transpose(1, 0, 2)
        valid[:, history - filled:] = True
        return x, valid


if __name__ == "__main__":
    from vehicles.entity.species import Fish
    from vehicles.learning.encoding import ObservationEncoder
    from vehicles.world.terrain import Barrier, Polygon, Terrain, Wall
    from vehicles.world.world import World

    world = World(seed=0)
    world.add_entity([Fish(position=(300 + 20 * i, 300), facing_point=(300 + 20 * i, 310)) for i in range(5)])
    world.add_terrain([
        Barrier(Wall([(200, 420), (450, 420)], thickness=8), sample_spacing=2.0),
        Terrain(Polygon.rectangle(center=(420, 220), size=(80, 40)), sample_spacing=2.0),
    ])

    spec = EncodingSpec()
    encoder = ObservationEncoder(Fish, world.entities, world.terrain, spec)
    history = History(spec, encoder.batch_size)

    for _ in range(20):
        world.step()
        history.record(*encoder.encode(world.frame))

    x, valid = history.batch()
    print(f"x {x.shape}, valid {valid.shape}, all valid: {bool(valid.all())}")
    print(encoder)
