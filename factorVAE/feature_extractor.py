import torch
import torch.nn as nn
import torch.nn.functional as F

from factorVAE.basic_net import MLP


class FeatureExtractor(nn.Module):
    def __init__(
        self, time_span, characteristic_size, latent_size, stock_size, gru_input_size
    ):
        """
        Generate latent features from historical sequential features.

        Args:
            time_span (int): T of data.
            characteristic_size (int): Size of characteristic.
            latent_size (int): Size of latent features.
            stock_size (int): Num of stocks.
            gru_input_size (int): Size of a hidden layers of GRU input.
        """
        super(FeatureExtractor, self).__init__()
        self.time_span = time_span
        self.characteristic_size = characteristic_size
        self.stock_size = stock_size
        self.gru_input_size = gru_input_size
        self.latent_size = latent_size

        self.proj = MLP(
            input_size=characteristic_size,
            output_size=gru_input_size,
            hidden_size=32,
            activation=nn.LeakyReLU(),
            out_activation=nn.LeakyReLU(),
        )
        self.gru = nn.GRU(input_size=gru_input_size, hidden_size=latent_size)

    def forward(self, x):
        """
        Generate latent features from historical sequential characteristics.

        Args:
            x (tensor): An array with the shape of (time_span, stock_size, characteristic_size)

        Returns:
            torch.tensor: The latent features of stocks with the shape of (1, stock_size, latent_size)
        """
        assert x.shape == (
            self.time_span,
            self.stock_size,
            self.characteristic_size,
        ), "input shape incorrect"

        h_proj = torch.zeros(
            (self.time_span, self.stock_size, self.gru_input_size),
            dtype=torch.float,
        )
        for i in range(self.time_span):
            for j in range(self.stock_size):
                h_proj[i, j] = self.proj(x[i, j])

        out, hidden = self.gru(h_proj)

        # e is the latent features of stock
        e = hidden.view((self.stock_size, self.latent_size))

        return e
