import torch
import torch.nn as nn

from factorVAE.basic_net import MLP


class FactorPredictor(nn.Module):
    def __init__(self, latent_size, factor_size, stock_size):
        super(FactorPredictor, self).__init__()

        self.multi_head_attention = nn.MultiheadAttention(latent_size, factor_size)

        self.distribution_network_mu = MLP(
            input_size=stock_size * latent_size, output_size=factor_size, hidden_size=64
        )

        self.distribution_network_sigma = MLP(
            input_size=stock_size * latent_size,
            output_size=factor_size,
            hidden_size=64,
            out_activation=nn.Softplus(),
        )

    def forward(self, latent_features):
        latent_features = latent_features.unsqueeze(1)
        h = self.multi_head_attention(
            latent_features, latent_features, latent_features
        )[0].flatten()

        mu_prior = self.distribution_network_mu(h)
        sigma_prior = self.distribution_network_sigma(h)

        return mu_prior, sigma_prior
