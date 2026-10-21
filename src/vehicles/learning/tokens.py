"""
One integer token per perceivable kind; the values filling the perception grid.

Values are contiguous from 0, so len(Token) is the embedding size. New members go at the end.
"""
from enum import IntEnum

from vehicles.entity.base_object import BaseObject
from vehicles.entity.environment import Grass, Rock, Tree
from vehicles.entity.species import Fish, Herbivore, Predator
from vehicles.entity.vehicle import Vehicle
from vehicles.world.terrain import Barrier, Terrain


class Token(IntEnum):
    EMPTY = 0
    VEHICLE = 1
    FISH = 2
    PREDATOR = 3
    HERBIVORE = 4
    OBJECT = 5
    ROCK = 6
    TREE = 7
    GRASS = 8
    TERRAIN = 9
    BARRIER = 10


KINDS = {
    Vehicle: Token.VEHICLE,
    Fish: Token.FISH,
    Predator: Token.PREDATOR,
    Herbivore: Token.HERBIVORE,
    BaseObject: Token.OBJECT,
    Rock: Token.ROCK,
    Tree: Token.TREE,
    Grass: Token.GRASS,
    Terrain: Token.TERRAIN,
    Barrier: Token.BARRIER,
}


def token_for(kind: type) -> Token:
    """
    Token of the nearest registered class in `kind`'s MRO.

    Raises
    ------
    KeyError : no class in the MRO has a token
    """
    for cls in kind.__mro__:
        if cls in KINDS:
            return KINDS[cls]
    raise KeyError(f"No token for {kind.__name__}")
