"""New experiments: stable first-order TSK with optional learned Gaussian premises.

Original SimpleANFIS remains in main.py. Float64 throughout; no test access here.
"""
import itertools
import numpy as np
from scipy.special import expit, logsumexp


class ResearchANFIS:
    def __init__(self, d, seed=42, trainable=False, lr=.01, l2=.001,
                 epochs=400, patience=50):
        self.d, self.seed, self.trainable = d, seed, trainable
        self.lr, self.l2, self.epochs, self.patience = lr, l2, epochs, patience
        self.rules = np.array(list(itertools.product(range(3), repeat=d)))
        self.history = []

    def initialize(self, X):
        self.centers = np.percentile(X, [25, 50, 75], axis=0).T.copy()
        widths = np.maximum(np.std(X, axis=0) * .75, .15)
        self.log_widths = np.log(np.repeat(widths[:, None], 3, axis=1))
        self.coef = np.random.default_rng(self.seed).normal(0, .05,
                                                          (len(self.rules), self.d + 1))

    def parts(self, X):
        widths = np.exp(self.log_widths)
        delta = X[:, :, None] - self.centers[None]
        log_mu = -.5 * (delta / widths[None]) ** 2
        log_w = sum(log_mu[:, j, self.rules[:, j]] for j in range(self.d))
        normalized = np.exp(log_w - logsumexp(log_w, axis=1, keepdims=True))
        xb = np.column_stack([X, np.ones(len(X))])
        f = xb @ self.coef.T
        z = (normalized * f).sum(axis=1)
        return log_mu, log_w, normalized, f, z

    def forward(self, X):
        return self.parts(X)[-1]

    def predict_proba(self, X):
        return expit(self.forward(X))

    def objective(self, X, y, weights, gradients=False):
        log_mu, log_w, a, f, z = self.parts(X)
        loss = np.sum(weights * (np.logaddexp(0, z) - y*z)) / weights.sum()
        loss += .5*self.l2*np.sum(self.coef**2)
        if not gradients:
            return float(loss)
        e = weights * (expit(z) - y) / weights.sum()
        xb = np.column_stack([X, np.ones(len(X))])
        gc = (a * e[:, None]).T @ xb + self.l2*self.coef
        gcenter, gwidth = np.zeros_like(self.centers), np.zeros_like(self.log_widths)
        if self.trainable:
            # dz/dlog(w_r) = a_r (f_r-z). Sum shared premise derivatives.
            q = e[:, None]*a*(f-z[:, None])
            widths2 = np.exp(2*self.log_widths)
            for j in range(self.d):
                for k in range(3):
                    shared = q[:, self.rules[:, j] == k].sum(axis=1)
                    delta = X[:, j]-self.centers[j, k]
                    gcenter[j, k] = np.sum(shared*delta/widths2[j, k])
                    gwidth[j, k] = np.sum(shared*delta**2/widths2[j, k])
        return float(loss), [gc, gcenter, gwidth]

    def fit(self, X, y, Xv, yv, weights=None, val_weights=None):
        self.initialize(X)
        weights = np.ones(len(y)) if weights is None else weights
        val_weights = np.ones(len(yv)) if val_weights is None else val_weights
        params = [self.coef, self.centers, self.log_widths]
        m, v = [np.zeros_like(p) for p in params], [np.zeros_like(p) for p in params]
        best, wait = np.inf, 0
        best_params = [p.copy() for p in params]
        for epoch in range(1, self.epochs+1):
            loss, grads = self.objective(X, y, weights, True)
            grad_norm = np.sqrt(sum(np.sum(g*g) for g in grads))
            for i, (p, g) in enumerate(zip(params, grads)):
                if i and not self.trainable:
                    continue
                m[i] = .9*m[i]+.1*g
                v[i] = .999*v[i]+.001*g*g
                p -= self.lr*(m[i]/(1-.9**epoch))/(np.sqrt(v[i]/(1-.999**epoch))+1e-8)
            np.clip(self.log_widths, np.log(.05), np.log(5.), out=self.log_widths)
            val = self.objective(Xv, yv, val_weights)
            self.history.append(dict(epoch=epoch, train_objective=loss,
                                     val_objective=val, gradient_norm=grad_norm))
            if val < best - 1e-6:
                best, wait, self.best_epoch = val, 0, epoch
                best_params = [p.copy() for p in params]
            else:
                wait += 1
            if wait >= self.patience:
                break
        self.coef, self.centers, self.log_widths = best_params
        return self
