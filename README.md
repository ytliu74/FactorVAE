# Reproduce AAAI22-FactorVAE

Issues will be attended to afterwards when I have more spare time. Happy and willing to receive more issues.

## Known Issues

1. In `factor_encoder.py`, the PortfolioLayer misrepresent the paper's idea of "a portfolio set". The current version only use "one portfolio" instead of "a set of".
2. Dataset now is not consistent with the paper due to some personal reason.
3. In `feature_extractor.py`, the latent feature, which is the GRU's output, seems quirky.
