import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size=32,
        activation=nn.LeakyReLU(),
        out_activation=nn.LeakyReLU(),
    ):
        """Generate a basic MLP

        Args:
            input_size (int)
            output_size (int)
            hidden_size (int, list, optional): Defaults to 32.
            activation (optional): Defaults to nn.LeakyReLU.
            out_activation (optional): Activation function of the output layer. Defaults to nn.LeakyReLU.
        """
        super(MLP, self).__init__()

        if type(hidden_size) is list:
            num_hidden_layer = len(hidden_size)
            self.net = nn.Sequential()
            self.net.add_module("input", nn.Linear(input_size, hidden_size[0]))
            for i in range(num_hidden_layer - 1):
                self.net.add_module(
                    f"hidden_{i}", nn.Linear(hidden_size[i], hidden_size[i + 1])
                )
            self.net.add_module(
                "out", nn.Linear(hidden_size[num_hidden_layer - 1], output_size)
            )
            self.net.add_module(
                "out_activ", out_activation
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                activation,
                nn.Linear(hidden_size, output_size),
                out_activation
            )

    def forward(self, x):
        out = self.net(x)

        return out
