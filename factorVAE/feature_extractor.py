import torch
import torch.nn as nn
import torch.nn.functional as F

from factorVAE.basic_net import MLP


class FeatureExtractor(nn.Module):
    def __init__(self, time_span, characteristic_size, latent_size, gru_input_size):
        """Feature Extractor

        Generate latent features from historical sequential features.

        Args:
            input_size (int): Input stock feature size
            output_size (int): Output latent feature size (e)
            gru_input_size (int): The input size of gru, which is the dimension H of the hidden states
        """
        super(FeatureExtractor, self).__init__()
        self.time_span = time_span
        self.input_size = characteristic_size

        self.proj = MLP(
            input_size=characteristic_size,
            output_size=gru_input_size,
            hidden_size=32,
            activation=nn.LeakyReLU(),
            out_activation=nn.LeakyReLU()
        )
        self.gru = nn.GRU(
            input_size=gru_input_size, hidden_size=latent_size, batch_first=True
        )

    def forward(self, x):
        """
        Args:
            x (array): An array with the shape of (time_span, input_size)
        """
        assert x.shape == (self.time_span, self.input_size), "input shape incorrect"
        h_proj = []
        for i in range(self.time_span):
            h_proj.append(self.proj(x[i]))

        h_proj = torch.stack(h_proj).unsqueeze(0)

        out, hidden = self.gru(h_proj)

        # e is the latent features of stock
        e = hidden

        return e
