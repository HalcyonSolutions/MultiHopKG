# Import Abstract Classes
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch


@dataclass
class Observation:
    state: torch.Tensor # Part of computation Graph
    kge_cur_pos: torch.Tensor # KG embedding vector of the current position
    kge_prev_pos: torch.Tensor # KG embedding vector of the previous position
    kge_action: torch.Tensor # KG embedding vector of the action

class Environment(ABC):

    @abstractmethod
    def reset(self, initial_states_info: torch.Tensor) -> Observation:
        """
        Args:
            - initial_state_info: any info that you wan to pass to initiazation.
        Returns:
            - position (torch.Tensor): Position in the graph
            - state (torch.Tensor): State containing informatioon for decision making.
        Both are meant to denote high abstraction so think about how to fit you are idea to them.
        """
        pass

    @abstractmethod
    def step(self, actions: torch.Tensor) -> Observation:
        """
        Args:
            - action (torch.Tensor): The action to take
        Returns:
            - position (torch.Tensor): Position in the graph
            - state (torch.Tensor): State containing informatioon for decision making.
            - retrun
        """
        pass


