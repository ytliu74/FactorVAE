import torch
import torch.nn as nn
from torch.distributions import Normal, kl_divergence

from factorVAE.feature_extractor import FeatureExtractor
from factorVAE.factor_encoder import FactorEncoder
from factorVAE.factor_decoder import FactorDecoder
from factorVAE.factor_predictor import FactorPredictor


class FactorVAE(nn.Module):
    def __init__(
        self,
        characteristic_size,
        stock_size,
        latent_size,
        factor_size,
        time_span,
        gru_input_size,
        hidden_size=64,
        alpha_h_size=64,
    ):
        super(FactorVAE, self).__init__()

        self.characteristic_size = characteristic_size
        self.stock_size = stock_size
        self.latent_size = latent_size

        self.feature_extractor = FeatureExtractor(
            time_span=time_span,
            characteristic_size=characteristic_size,
            latent_size=latent_size,
            stock_size=stock_size,
            gru_input_size=gru_input_size,
        )

        self.factor_encoder = FactorEncoder(
            latent_size=latent_size,
            stock_size=stock_size,
            factor_size=factor_size,
            hidden_size=hidden_size,
        )

        self.factor_decoder = FactorDecoder(
            latent_size=latent_size,
            factor_size=factor_size,
            stock_size=stock_size,
            alpha_h_size=alpha_h_size,
            hidden_size=hidden_size,
        )

        self.factor_predictor = FactorPredictor(
            latent_size=latent_size, factor_size=factor_size, stock_size=stock_size
        )

    def train_model(self, characteristics, future_returns, gamma=1):
        latent_features = self.feature_extractor(characteristics)

        mu_post, sigma_post = self.factor_encoder(latent_features, future_returns)
        m_encoder = Normal(mu_post, sigma_post)
        factors_post = m_encoder.sample()

        reconstruct_returns, mu_alpha, sigma_alpha, beta = self.factor_decoder(
            factors_post, latent_features
        )

        loss_negloglike = 0
        mu_dec, sigma_dec = self.get_decoder_distribution(mu_alpha, sigma_alpha, mu_post, sigma_post, beta)
        for i in range(self.stock_size):
            loss_negloglike += Normal(mu_dec[i], sigma_dec[i]).log_prob(
                future_returns[i]
            )
        loss_negloglike = loss_negloglike * (-1 / self.stock_size)

        mu_prior, sigma_prior = self.factor_predictor(latent_features)
        m_predictor = Normal(mu_prior, sigma_prior)

        loss_KL = kl_divergence(m_encoder, m_predictor)

        loss = loss_negloglike + gamma * loss_KL

        return loss

    def get_decoder_distribution(
        self, mu_alpha, sigma_alpha, mu_post, sigma_post, beta
    ):
        mu_dec = mu_alpha + torch.mm(mu_post, beta)

        sigma_dec = torch.sqrt(
            torch.square(sigma_alpha)
            + torch.mm(torch.square(sigma_post), torch.square(beta))
        )

        return mu_dec.squeeze(0), sigma_dec.squeeze(0)
