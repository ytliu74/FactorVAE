import torch
import torch.nn as nn
import torch.nn.functional as F

from factorVAE.basic_net import MLP
from torch.distributions import Normal


class FactorDecoder(nn.Module):
    def __init__(
        self, latent_size, factor_size, stock_size, alpha_h_size=64, hidden_size=64
    ):
        super(FactorDecoder, self).__init__()
        self.alpha_layer = AlphaLayer(
            latent_size=latent_size,
            h_size=alpha_h_size,
            hidden_size=hidden_size,
            stock_size=stock_size,
        )
        self.beta_layer = BetaLayer(
            latent_size=latent_size,
            stock_size=stock_size,
            factor_size=factor_size,
            hidden_size=hidden_size
        )

    def forward(self, factors, latent_features):
        alpha = self.alpha_layer(latent_features)
        beta = self.beta_layer(latent_features)

        exposed_factors = torch.mm(factors, beta)

        stock_returns = exposed_factors + alpha

        return stock_returns


class AlphaLayer(nn.Module):
    def __init__(self, latent_size, h_size, hidden_size, stock_size):
        super(AlphaLayer, self).__init__()
        self.hidden_layer = MLP(
            input_size=latent_size,
            output_size=h_size,
            hidden_size=hidden_size,
            activation=nn.LeakyReLU(),
            out_activation=nn.LeakyReLU(),
        )
        self.alpha_mu_layer = MLP(
            input_size=h_size,
            output_size=stock_size,
            hidden_size=hidden_size,
            activation=nn.LeakyReLU(),
            out_activation=nn.LeakyReLU(),
        )
        self.alpha_sigma_layer = MLP(
            input_size=h_size,
            output_size=stock_size,
            hidden_size=hidden_size,
            activation=nn.LeakyReLU(),
            out_activation=nn.Softplus(),
        )

    def forward(self, latent_features):
        hidden_state = self.hidden_layer(latent_features)
        alpha_mu = self.alpha_sigma_layer(hidden_state)
        alpha_sigma = self.alpha_sigma_layer(hidden_state)

        m = Normal(alpha_mu, alpha_sigma)
        alpha = m.sample()

        return alpha


class BetaLayer(nn.Module):
    def __init__(self, latent_size, stock_size, factor_size, hidden_size=64):
        super(BetaLayer, self).__init__()

        self.stock_size = stock_size
        self.factor_size = factor_size

        self.beta_layer = MLP(
            input_size=latent_size,
            output_size=stock_size * factor_size,
            hidden_size=hidden_size,
            activation=nn.LeakyReLU(),
            out_activation=nn.LeakyReLU(),
        )

    def forward(self, x):
        out = self.beta_layer(x)
        beta = out.view(self.factor_size, self.stock_size)

        return beta
