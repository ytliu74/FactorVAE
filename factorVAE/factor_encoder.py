from pyrsistent import m
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Normal
from factorVAE.basic_net import MLP


class FactorEncoder(nn.Module):
    def __init__(self, latent_size, stock_size, factor_size, hidden_size=32):
        """ Factor Encoder

        Return mu_post, sigma_post

        Args:
            latent_size (int)
            stock_size (int)
            factor_size (int)
            hidden_size (int or list)
        """
        super(FactorEncoder, self).__init__()

        self.portfolio_layer = PortfolioLayer(latent_size, stock_size, hidden_size)
        self.mapping_layer = MappingLayer(stock_size, factor_size, hidden_size)

    def forward(self, latent_features, future_returns):
        portfolio_weights = self.portfolio_layer(latent_features)
        portfolio_returns = portfolio_weights * future_returns

        mu_post, sigma_post = self.mapping_layer(portfolio_returns)
        mu_post = mu_post.unsqueeze(-1)
        sigma_post = sigma_post.unsqueeze(-1)

        # m = Normal(mu_post, sigma_post)
        # z_post = m.sample()

        return mu_post, sigma_post


class MappingLayer(nn.Module):
    def __init__(self, stock_size, factor_size, hidden_size=16):
        super(MappingLayer, self).__init__()

        self.mu_net = MLP(
            input_size=stock_size,
            output_size=factor_size,
            hidden_size=hidden_size,
            activation=nn.LeakyReLU(),
            out_activation=nn.LeakyReLU()
        )

        self.sigma_net = MLP(
            input_size=stock_size,
            output_size=factor_size,
            hidden_size=hidden_size,
            activation=nn.LeakyReLU(),
            out_activation=nn.Softplus()
        )

    def forward(self, portfolio_returns):
        # portfolio_returns.shape =  (batch_size, stock_size)
        mu_post = self.mu_net(portfolio_returns)
        sigma_post = self.sigma_net(portfolio_returns)

        return mu_post, sigma_post


class PortfolioLayer(nn.Module):
    def __init__(self, latent_size, stock_size, hidden_size=32):
        super(PortfolioLayer, self).__init__()

        self.net = MLP(
            input_size=latent_size,
            output_size=1,
            hidden_size=hidden_size
        )

    def forward(self, latent_features):
        out = self.net(latent_features)

        out = torch.softmax(out, dim=1).squeeze(-1)
        # out.shape = (batch_size, stock_size)

        return out