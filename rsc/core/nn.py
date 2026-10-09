"""NN baseline wrapper. Seed = sha256(description) mod 2**31 (as in the original ImprovedNN).
Uses torch MLP (LayerNorm+SiLU, AdamW, cosine warm restarts) when torch is installed, otherwise a scikit-learn MLP
so smoke tests run anywhere. The paper's numbers must come from the torch path: `backend` is recorded."""
from __future__ import annotations
import hashlib, numpy as np

def seed_from(description: str) -> int:
    return int(hashlib.sha256(description.encode()).hexdigest(), 16) % (2 ** 31)

class NNBaseline:
    def __init__(self, description: str, epochs: int = 400, hidden: int = 64):
        self.seed, self.epochs, self.hidden = seed_from(description), epochs, hidden
        try:
            import torch; self.backend = "torch"
        except ImportError:
            self.backend = "sklearn"
    def fit(self, X, y):
        X = np.asarray(X, float); y = np.asarray(y, float).ravel()
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9; self.ym, self.ys = y.mean(), y.std() + 1e-9
        Xn, yn = (X - self.mu) / self.sd, (y - self.ym) / self.ys
        if self.backend == "torch":
            import torch, torch.nn as nn
            torch.manual_seed(self.seed)
            self.m = nn.Sequential(nn.Linear(X.shape[1], self.hidden), nn.LayerNorm(self.hidden), nn.SiLU(),
                                   nn.Linear(self.hidden, self.hidden), nn.LayerNorm(self.hidden), nn.SiLU(), nn.Linear(self.hidden, 1))
            opt = torch.optim.AdamW(self.m.parameters(), 3e-3); sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, 100)
            xt, yt = torch.tensor(Xn, dtype=torch.float32), torch.tensor(yn, dtype=torch.float32)[:, None]
            for _ in range(self.epochs):
                opt.zero_grad(); nn.functional.mse_loss(self.m(xt), yt).backward(); opt.step(); sch.step()
        else:
            from sklearn.neural_network import MLPRegressor
            self.m = MLPRegressor(hidden_layer_sizes=(self.hidden, self.hidden), max_iter=self.epochs * 3, random_state=self.seed % (2 ** 31)).fit(Xn, yn)
        return self
    def predict(self, X):
        Xn = (np.asarray(X, float) - self.mu) / self.sd
        if self.backend == "torch":
            import torch
            with torch.no_grad(): p = self.m(torch.tensor(Xn, dtype=torch.float32)).numpy().ravel()
        else:
            p = self.m.predict(Xn)
        return p * self.ys + self.ym
