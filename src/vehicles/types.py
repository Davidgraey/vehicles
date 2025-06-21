"""
Typing complex objects & definitions for __package__
"""
from enum import Enum, auto
import numpy as np
from abc import ABC, abstractmethod
from typing import Protocol, Tuple, Dict, List, Optional, Union, Iterable, Callable
from typing_extensions import TypedDict


# -------------- Model Base Class  --------------
class BasalModel(ABC):
    def __init__(self,
                 input_dimension: int,
                 output_dimension: int,
                 seed: int = 42):
        self.RNG = np.random.default_rng(seed)

        self.weights = np.empty()
        self.weights_shape: tuple[int, ...] = ()

        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        pass

    @abstractmethod
    def init_weights(self):
       # set up initial weights
       # assign self.weights_shape
       pass

    @abstractmethod
    def forward(self):
       pass

    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def calculate_gradients(self):
        pass

    @abstractmethod
    def calculate_loss(self):
        pass

    @property
    def weights(self):
        return self.weights[1:, ...]

    @property
    def weights(self):
        return self.weights[1:, ...]

# -------------- Complex typing  --------------
class ComplexType(TypedDict):
    attribute_1: int
    # ...

