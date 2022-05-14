import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    def __init__(
        self,
        input_size: int,
        n_actions: int,
        seed: int,
        hidden_size_1: int = 64,
        hidden_size_2: int = 64,
        hidden_size_3: int = 64,
    ):
        """
        Create a model.
        Params
        ======
            input_size (int): Dimension of each state
            n_actions (int): Dimension of each action
            seed (int): Random seed
            hidden_size_1 (int): Number of nodes in first hidden layer
            hidden_size_2 (int): Number of nodes in second hidden layer
        """
        super().__init__()
        self.seed = torch.manual_seed(seed)
        self.input_layer = nn.Linear(input_size, hidden_size_1)
        self.hidden_1 = nn.Linear(hidden_size_1, hidden_size_2)
        self.hidden_2 = nn.Linear(hidden_size_2, hidden_size_3)
        self.output_layer = nn.Linear(hidden_size_3, n_actions)

    def forward(self, x) -> torch.Tensor:
        x = F.relu(self.input_layer(x))
        x = F.relu(self.hidden_1(x))
        x = F.relu(self.hidden_2(x))
        return F.relu(self.output_layer(x))
