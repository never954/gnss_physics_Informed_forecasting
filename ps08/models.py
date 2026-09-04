"""Forecast models. Common interface:

    m = Model(period_h)
    m.fit(t_hours, y)          # 1-D continuous time -> values
    yhat = m.predict(t_hours)  # values at arbitrary query times

All models are per-series and per-channel: they fit on one satellite's own history
and extrapolate to arbitrary (irregular) day-8 timestamps. This makes them immune
to train/test distribution shift, which matters because the evaluator's satellites
differ from ours.
"""
import numpy as np
from scipy.stats import shapiro
from scipy.optimize import minimize
from sklearn.linear_model import HuberRegressor, TheilSenRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import (
    RBF, ExpSineSquared, WhiteKernel, ConstantKernel as C,
)

from outliers import mad_clip
from config import MAD_CLIP_THRESH, RANDOM_SEED


def _harmonic_features(t, period):
    """Design matrix (no intercept): trend + orbital + daily harmonics."""
    t = np.asarray(t, dtype=float)
    return np.column_stack([
        t, t ** 2,
        np.sin(2 * np.pi * t / period), np.cos(2 * np.pi * t / period),
        np.sin(2 * np.pi * t / 24.0),   np.cos(2 * np.pi * t / 24.0),
    ])


class CentralBaseline:
    """B0 — robust constant level (median of MAD-clipped training values).

    The floor: since day-ahead forecastability is ~0, this asks 'how good is
    predicting a stable centre?' and its residual is the raw noise itself.
    """
    name = "B0_robust_central"

    def __init__(self, period=None):
        self.level = 0.0

    def fit(self, t, y):
        self.level = float(np.median(mad_clip(y, MAD_CLIP_THRESH)))
        return self

    def predict(self, t):
        return np.full(len(np.asarray(t)), self.level)


class RobustHarmonic:
    """A1 — Huber-regressed trend + orbital + daily harmonics.

    Huber loss makes the fit robust to outliers without discarding points, so the
    model captures whatever weak deterministic structure exists while ignoring spikes.
    """
    name = "A1_robust_harmonic"

    def __init__(self, period):
        self.period = period

    def fit(self, t, y):
        X = _harmonic_features(t, self.period)
        self.scaler = StandardScaler().fit(X)
        self.model = HuberRegressor(alpha=1e-3, epsilon=1.35, max_iter=1000)
        self.model.fit(self.scaler.transform(X), np.asarray(y, dtype=float))
        return self

    def predict(self, t):
        X = self.scaler.transform(_harmonic_features(t, self.period))
        return self.model.predict(X)


class GPModel:
    """A2 — lean Gaussian Process: smooth trend + one learnable orbital period + noise.

    Continuous-time input eats irregular/arbitrary timestamps natively; the posterior
    mean is smooth and the output is Gaussian by construction. Training values are
    MAD-clipped first so the kernel is not distorted by spikes. Kept deliberately lean
    (one RBF + one periodic + white) to avoid overfitting 46-143 points.
    """
    name = "A2_gp"

    def __init__(self, period):
        self.period = period

    def fit(self, t, y):
        X = np.asarray(t, dtype=float).reshape(-1, 1)
        yc = mad_clip(y, thresh=4.0)
        span = float(X.max() - X.min()) + 1e-9
        kernel = (
            C(1.0, (1e-3, 1e3)) * RBF(span / 3.0, (span / 20.0, span * 3.0))
            + C(1.0, (1e-3, 1e3)) * ExpSineSquared(
                length_scale=self.period, periodicity=self.period,
                periodicity_bounds=(self.period * 0.7, self.period * 1.3))
            + WhiteKernel(np.var(yc) + 1e-6, (1e-6, 1e6))
        )
        self.gp = GPR(kernel=kernel, normalize_y=True,
                      n_restarts_optimizer=3, alpha=1e-6,
                      random_state=RANDOM_SEED)
        self.gp.fit(X, yc)
        return self

    def predict(self, t):
        return self.gp.predict(np.asarray(t, dtype=float).reshape(-1, 1))


class KalmanLLT:
    """A3 — continuous-time local linear trend (integrated random walk) Kalman filter.

    State = [level, slope]; the transition depends on the gap dt between consecutive
    (irregular) timestamps, so no resampling is needed. Two hyperparameters — process
    noise q and observation noise r — are fit by maximum likelihood. Suits GEO's
    random-walk character: the MLE shrinks the slope when the data won't support a trend,
    collapsing gracefully toward a robust level.
    """
    name = "A3_kalman"

    def __init__(self, period=None):
        pass

    @staticmethod
    def _filter(t, y, q, r):
        """Run the filter; return (neg_loglik, final_state, final_time)."""
        x = np.array([y[0], 0.0])
        P = np.eye(2) * 1e3  # diffuse prior
        H = np.array([1.0, 0.0])
        nll = 0.0
        for i in range(1, len(t)):
            dt = t[i] - t[i - 1]
            if dt <= 0:
                dt = 1e-6
            F = np.array([[1.0, dt], [0.0, 1.0]])
            Q = q * np.array([[dt ** 3 / 3.0, dt ** 2 / 2.0],
                              [dt ** 2 / 2.0, dt]])
            x = F @ x
            P = F @ P @ F.T + Q
            S = P[0, 0] + r
            innov = y[i] - x[0]
            K = (P @ H) / S
            x = x + K * innov
            P = P - np.outer(K, H @ P)
            nll += 0.5 * (np.log(2.0 * np.pi * S) + innov ** 2 / S)
        return nll, x, t[-1]

    def fit(self, t, y):
        t = np.asarray(t, dtype=float)
        order = np.argsort(t)
        t = t[order]
        y = mad_clip(np.asarray(y, dtype=float)[order])
        self.mu = float(np.median(y))
        self.sd = float(np.std(y)) + 1e-9
        yn = (y - self.mu) / self.sd

        def obj(params):
            q, r = np.exp(params)
            try:
                return self._filter(t, yn, q, r)[0]
            except Exception:
                return 1e12

        res = minimize(obj, x0=np.log([1e-3, 1e-1]), method="Nelder-Mead",
                       options={"maxiter": 300, "xatol": 1e-2, "fatol": 1e-2})
        self.q, self.r = np.exp(res.x)
        _, self.state, self.t_last = self._filter(t, yn, self.q, self.r)
        return self

    def predict(self, t):
        t = np.asarray(t, dtype=float)
        level, slope = self.state
        pred_n = level + slope * (t - self.t_last)
        return pred_n * self.sd + self.mu


class EnsembleSelector:
    """C1 — per-channel model selection by held-out Shapiro-W.

    Holds out the tail of the training series (time-ordered), scores each base model's
    residual normality on it, keeps the winner, and refits it on the full series. This
    lets GEO route to a robust/level model and MEO to a smoother one without hand-coding.
    """
    name = "C1_ensemble"
    BASE = [CentralBaseline, RobustHarmonic, GPModel, KalmanLLT]

    def __init__(self, period):
        self.period = period

    def fit(self, t, y):
        t = np.asarray(t, dtype=float)
        y = np.asarray(y, dtype=float)
        order = np.argsort(t)
        t, y = t[order], y[order]
        k = max(4, int(round(len(t) * 0.3)))
        t_fit, y_fit, t_val, y_val = t[:-k], y[:-k], t[-k:], y[-k:]

        best_cls, best_W = CentralBaseline, -np.inf
        self.scores = {}
        for cls in self.BASE:
            try:
                m = cls(self.period).fit(t_fit, y_fit)
                res = np.asarray(m.predict(t_val), float) - y_val
                W = shapiro(res)[0] if len(res) >= 3 else -np.inf
            except Exception:
                W = -np.inf
            self.scores[cls.name] = W
            if W > best_W:
                best_W, best_cls = W, cls
        self.choice = best_cls.name
        self.model = best_cls(self.period).fit(t, y)  # refit on full series
        return self

    def predict(self, t):
        return self.model.predict(t)


class ComposedPipeline:
    """P1 — the recommended composition (not a vote): each stage solves one sub-problem.

        Stage 1  robust detrend    — Theil-Sen linear fit removes gross drift robustly
        Stage 2  light outlier clip — MAD-clip the detrended residual (done inside the GP)
        Stage 3  lean GP           — model the remaining structure + noise in continuous time
        (Stage 4 validated Gaussian reporting lives in report.py, not in the model)

    Prediction = extrapolated robust trend + GP posterior mean. Deliberately conservative:
    the Theil-Sen trend resists outliers, and the GP reverts to the residual mean far from data.
    """
    name = "P1_composed"

    def __init__(self, period):
        self.period = period

    def fit(self, t, y):
        t = np.asarray(t, dtype=float)
        y = np.asarray(y, dtype=float)
        # Stage 1: robust linear detrend
        self.trend = TheilSenRegressor(random_state=RANDOM_SEED, max_iter=500)
        self.trend.fit(t.reshape(-1, 1), y)
        resid = y - self.trend.predict(t.reshape(-1, 1))
        # Stages 2+3: lean GP (its own light MAD clip) on the detrended residual
        self.gp = GPModel(self.period).fit(t, resid)
        return self

    def predict(self, t):
        t = np.asarray(t, dtype=float)
        return self.trend.predict(t.reshape(-1, 1)) + self.gp.predict(t)


class BiasCorrected:
    """Post-process wrapper: remove systematic prediction bias (Priority-2 optimisation).

    Shapiro-W is shift-invariant, so centering the residual at zero never changes the
    Priority-1 score — but it improves Priority-2 (mean, and std where the error was
    bias-dominated). The bias is estimated on a time-ordered training hold-out (the tail,
    which is closest to the forecast region), then subtracted from all predictions.
    Uses only training data — never the test truth.
    """
    def __init__(self, base_cls, period, holdout_frac=0.3):
        self.base_cls = base_cls
        self.period = period
        self.holdout_frac = holdout_frac
        self.bias = 0.0

    def fit(self, t, y):
        t = np.asarray(t, dtype=float)
        y = np.asarray(y, dtype=float)
        order = np.argsort(t)
        t, y = t[order], y[order]
        n = len(t)
        k = max(3, int(round(n * self.holdout_frac)))
        if n - k >= 3:
            base = self.base_cls(self.period).fit(t[:-k], y[:-k])
            self.bias = float(np.mean(np.asarray(base.predict(t[-k:]), float) - y[-k:]))
        else:
            self.bias = 0.0
        self.model = self.base_cls(self.period).fit(t, y)
        return self

    def predict(self, t):
        return np.asarray(self.model.predict(t), float) - self.bias


# Registry: name -> factory(period) -> model
MODELS = {
    CentralBaseline.name: lambda period: CentralBaseline(period),
    RobustHarmonic.name:  lambda period: RobustHarmonic(period),
    GPModel.name:         lambda period: GPModel(period),
    KalmanLLT.name:       lambda period: KalmanLLT(period),
    EnsembleSelector.name: lambda period: EnsembleSelector(period),
    ComposedPipeline.name: lambda period: ComposedPipeline(period),
    "A2_gp_bc":           lambda period: BiasCorrected(GPModel, period),
    "P1_composed_bc":     lambda period: BiasCorrected(ComposedPipeline, period),
}
