"""Environmental entities: static barriers and food-producing plants."""

from vehicles.entity.base_object import BaseObject


class Rock(BaseObject):
    """
    Static barrier with large mass, so collisions barely move it.

    Parameters
    ----------
    position : tuple of int
    facing_point : tuple of int
    mass : int, default 50
    size : tuple of int, default (30, 30)
    """

    def __init__(self, position, facing_point, mass=5000, size=(30, 30)):
        super().__init__(mass=mass, position=position, size=size, facing_point=facing_point)
        self.color = "dimgray"


class Plant(BaseObject):
    """
    Stationary food source that replenishes over time.

    Parameters
    ----------
    position : tuple of int
    facing_point : tuple of int
    mass : int
    size : tuple of int
    growth_rate : float
        Food regenerated per tick.
    max_food : float
        Capacity: the most food this plant can hold at once.
    """

    def __init__(self, position, facing_point, mass, size, growth_rate, max_food):
        super().__init__(mass=mass, position=position, size=size, facing_point=facing_point)
        self.growth_rate = growth_rate
        self.max_food = max_food
        self.food = max_food

    def grow(self):
        """Replenish food by growth_rate, capped at max_food."""
        self.food = min(self.food + self.growth_rate, self.max_food)

    def consume(self, amount):
        """
        Remove food, floored at 0.

        Parameters
        ----------
        amount : float

        Returns
        -------
        float
            Amount actually consumed.
        """
        consumed = min(amount, self.food)
        self.food -= consumed
        return consumed


class Tree(Plant):
    """High mass and size, slow growth, large capacity."""

    def __init__(self, position, facing_point, mass=10000, size=(40, 40),
                 growth_rate=0.1, max_food=500):
        super().__init__(position=position, facing_point=facing_point,
                          mass=mass, size=size,
                          growth_rate=growth_rate, max_food=max_food)
        self.color = "darkgreen"


class Grass(Plant):
    """Low mass and size, fast growth, small capacity."""

    def __init__(self, position, facing_point, mass=5000, size=(6, 6),
                 growth_rate=0.5, max_food=10.0):
        super().__init__(position=position, facing_point=facing_point,
                          mass=mass, size=size,
                          growth_rate=growth_rate, max_food=max_food)
        self.color = "yellowgreen"
