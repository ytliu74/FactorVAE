import torch
import torch.nn as nn
import torch.nn.functional as F

from factorVAE.basic_net import MLP
from torch.distributions import Normal


class FactorDecoder(nn.Module):
    def __init__(
        self, latent_size, factor_size, stock_size, alpha_h_size=64, hidden_size=64
    ):
        """Generate Stock return y hat from factors ang latent features.

        Args:
            latent_size (int)
            factor_size (int)
            stock_size (int)
            alpha_h_size (int): The size of the hidden layer in alpha layer. Defaults to 64.
            hidden_size (int or list): Defaults to 64.
        """
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
        mu_alpha, sigma_alpha = self.alpha_layer(latent_features)
        m_alpha = Normal(mu_alpha, sigma_alpha)
        alpha = m_alpha.sample()

        beta = self.beta_layer(latent_features)

        exposed_factors = torch.bmm(beta, factors)

        stock_returns = exposed_factors + alpha

        return stock_returns, mu_alpha, sigma_alpha, beta


class AlphaLayer(nn.Module):
    def __init__(self, latent_size, h_size, hidden_size, stock_size):
        super(AlphaLayer, self).__init__()
        self.h_size = h_size
        self.stock_size = stock_size

        self.hidden_layer = MLP(
            input_size=latent_size,
            output_size=h_size,
            hidden_size=hidden_size,
            activation=nn.LeakyReLU(),
            out_activation=nn.LeakyReLU(),
        )
        self.mu_alpha_layer = MLP(
            input_size=h_size,
            output_size=1,
            hidden_size=hidden_size,
            activation=nn.LeakyReLU(),
            out_activation=nn.LeakyReLU(),
        )
        self.sigma_alpha_layer = MLP(
            input_size=h_size,
            output_size=1,
            hidden_size=hidden_size,
            activation=nn.LeakyReLU(),
            out_activation=nn.Softplus(),
        )

    def forward(self, latent_features):
        # Input(latent_features) shape = (batch_size, stock_size, latent_size)
        hidden_state = self.hidden_layer(latent_features)
        mu_alpha = self.mu_alpha_layer(hidden_state)
        sigma_alpha = self.sigma_alpha_layer(hidden_state)
        # (batch_size, stock_size, 1)

        return mu_alpha, sigma_alpha


class BetaLayer(nn.Module):
    def __init__(self, latent_size, stock_size, factor_size, hidden_size=64):
        super(BetaLayer, self).__init__()

        self.factor_size = factor_size
        self.stock_size = stock_size

        self.beta_layer = MLP(
            input_size=latent_size,
            output_size=factor_size,
            hidden_size=hidden_size,
            activation=nn.LeakyReLU(),
            out_activation=nn.LeakyReLU(),
        )

    def forward(self, latent_features):
        # (bs, stock_size, latent_size) -> (bs, stock_size, factor_size)
        beta = self.beta_layer(latent_features)

        return beta
