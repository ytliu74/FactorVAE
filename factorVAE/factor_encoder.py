import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class FactorEncoder(nn.Module):
    def __init__(self, latent_size, stock_size, factor_size, hidden_size=16):
        super(FactorEncoder, self).__init__()

        self.portfolio_layer = PortfolioLayer(latent_size, stock_size, hidden_size)
        self.mapping_layer = MappingLayer(stock_size, factor_size, hidden_size)

    def forward(self, latent_features, future_returns):
        portfolio_weights = self.portfolio_layer(latent_features)
        portfolio_returns = portfolio_weights * future_returns

        mu_post, sigma_post = self.mapping_layer(portfolio_returns)

        m = Normal(mu_post, sigma_post)
        factor = m.sample()

        return factor


class MappingLayer(nn.Module):
    def __init__(self, stock_size, factor_size, hidden_size=16):
        super(MappingLayer, self).__init__()

        self.mu_net = nn.Sequential(
            nn.Linear(stock_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, factor_size),
        )

        self.sigma_net = nn.Sequential(
            nn.Linear(stock_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, factor_size),
            nn.Softplus(beta=1),
        )

    def forward(self, x):
        mu_post = self.mu_net(x)
        sigma_post = self.sigma_net(x)

        return mu_post, sigma_post


class PortfolioLayer(nn.Module):
    def __init__(self, latent_size, stock_size, hidden_size=16):
        super(PortfolioLayer, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, stock_size),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        out = self.net(x)

        return out

if __name__ == '__main__':
    feat = torch.tensor([1, 2, 3, 4], dtype=torch.float)
    ret = torch.tensor([1, 2], dtype=torch.float)
    fe = FactorEncoder(latent_size=4, stock_size=2, factor_size=3)
    print(fe(feat, ret))