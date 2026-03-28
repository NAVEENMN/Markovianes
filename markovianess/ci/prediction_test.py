"""
prediction_test.py
------------------
Prediction-based Markov violation score.

Compares prediction accuracy with and without observation history:
  1. Markov model:  f(o_t) -> o_{t+1}           (current observation only)
  2. History model:  g(o_t, o_{t-1}, ..., o_{t-k}) -> o_{t+1}  (uses past observations)
  3. MVS = (MSE_markov - MSE_history) / MSE_markov, clipped to [0, 1]

Approach:
  - RF-residual Ridge: Random Forest on current obs removes the nonlinear
    Markov component via OOB predictions. Then RidgeCV on the residuals
    checks if history helps predict what current obs cannot.
    This eliminates false positives from nonlinear dynamics (Pendulum,
    Acrobot) because the RF captures nonlinearity from o_t alone.
  - NN stage (disabled by default): Neural network with group dropout
    for nonlinear history dependence. Currently produces false positives
    on trained-policy observations due to overfitting to structured
    trajectories. Enable with use_nn=True if needed.

If Markov: history doesn't help -> MVS ≈ 0.
If non-Markov: history helps -> MVS > 0.
"""

import logging

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger("markovianess")


class _PredictionMLP(nn.Module):
    """Simple feedforward network for next-observation prediction."""

    def __init__(self, input_dim, output_dim, hidden_size=64, n_hidden=2):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_size), nn.ReLU()]
        for _ in range(n_hidden - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.ReLU()]
        layers.append(nn.Linear(hidden_size, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class PredictionMarkovTest:
    """
    Prediction-based Markov violation test.

    Uses a two-stage approach:
    1. RF-residual Ridge: RF removes nonlinear Markov dynamics, then Ridge
       on residuals checks if history carries additional predictive signal.
    2. Neural network with group dropout (captures nonlinear effects)

    Output is compatible with existing CI test backends — .run() returns a dict
    with p_matrix, val_matrix, and lag2_or_more_links keys.
    """

    def __init__(self, history_len=3, n_runs=3, hidden_size=64, n_hidden=2,
                 batch_size=256, lr=1e-3, weight_decay=1e-4,
                 max_epochs=200, patience=15, p_mask=0.5,
                 use_nn=False):
        self.history_len = history_len
        self.n_runs = n_runs
        self.hidden_size = hidden_size
        self.n_hidden = n_hidden
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.patience = patience
        self.p_mask = p_mask
        self.use_nn = use_nn
        self.results = None

    @staticmethod
    def _build_datasets(observations, history_len):
        """
        Build input matrices for next-step prediction.

        Returns
        -------
        X_markov : np.ndarray, shape (samples, N)
        X_history : np.ndarray, shape (samples, N * (k + 1))
        Y : np.ndarray, shape (samples, N)
        """
        T, N = observations.shape
        k = history_len

        n_samples = T - k - 1
        if n_samples <= 0:
            raise ValueError(
                f"Not enough data: T={T}, history_len={k}, need at least {k + 2} steps"
            )

        X_markov = np.empty((n_samples, N), dtype=np.float32)
        X_history = np.empty((n_samples, N * (k + 1)), dtype=np.float32)
        Y = np.empty((n_samples, N), dtype=np.float32)

        for idx, t in enumerate(range(k, T - 1)):
            X_markov[idx] = observations[t]
            # [o_t, o_{t-1}, ..., o_{t-k}]
            history = observations[t - k : t + 1][::-1]
            X_history[idx] = history.flatten()
            Y[idx] = observations[t + 1]

        return X_markov, X_history, Y

    # ------------------------------------------------------------------
    # Stage 1: RF-residual Ridge
    # ------------------------------------------------------------------

    @staticmethod
    def _ridge_mvs(X_m_train, X_m_test, X_h_train, X_h_test, Y_train, Y_test):
        """
        Compute MVS using RF-residual Ridge.

        Two-stage approach:
          1. RF on X_markov -> Y removes the nonlinear Markov component.
             OOB predictions provide honest training residuals.
          2. Ridge comparison on residuals checks if history helps predict
             what the current observation cannot.

        This eliminates false positives from nonlinear dynamics because
        the RF captures nonlinearity from o_t alone, leaving only noise
        + genuine history-dependent signal in the residuals.
        """
        # Stage 1: Remove Markov component with RF
        # n_jobs=1 to avoid oversubscription when called from parallel workers
        rf = RandomForestRegressor(
            n_estimators=200, max_depth=10, min_samples_leaf=5,
            oob_score=True, random_state=42, n_jobs=1,
        )
        rf.fit(X_m_train, Y_train)

        # OOB residuals for training (honest, not overfit)
        residuals_train = Y_train - rf.oob_prediction_
        # Test residuals
        residuals_test = Y_test - rf.predict(X_m_test)

        # Stage 2: Ridge comparison on residuals
        alphas = np.logspace(-2, 5, 20)

        # Markov Ridge: can current obs predict residuals?
        ridge_m = RidgeCV(alphas=alphas)
        ridge_m.fit(X_m_train, residuals_train)
        pred_m = ridge_m.predict(X_m_test)
        mse_m = float(np.mean((residuals_test - pred_m) ** 2))

        # History Ridge: can history help predict residuals?
        ridge_h = RidgeCV(alphas=alphas)
        ridge_h.fit(X_h_train, residuals_train)
        pred_h = ridge_h.predict(X_h_test)
        mse_h = float(np.mean((residuals_test - pred_h) ** 2))

        if mse_m < 1e-15:
            mvs = 0.0
        else:
            mvs = (mse_m - mse_h) / mse_m
            mvs = float(np.clip(mvs, 0.0, 1.0))

        return mvs, mse_m, mse_h

    # ------------------------------------------------------------------
    # Stage 2: Neural network with group dropout
    # ------------------------------------------------------------------

    def _train_nn(self, X_train, Y_train, obs_dim, seed=0):
        """Train MLP with group dropout on history features."""
        torch.manual_seed(seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_dim = X_train.shape[1]
        output_dim = Y_train.shape[1]

        X_mean = X_train.mean(axis=0)
        X_std = X_train.std(axis=0) + 1e-8
        Y_mean = Y_train.mean(axis=0)
        Y_std = Y_train.std(axis=0) + 1e-8

        X_train_n = (X_train - X_mean) / X_std
        Y_train_n = (Y_train - Y_mean) / Y_std

        n = len(X_train_n)
        n_tr = int(0.85 * n)
        X_tr = torch.tensor(X_train_n[:n_tr], dtype=torch.float32, device=device)
        Y_tr = torch.tensor(Y_train_n[:n_tr], dtype=torch.float32, device=device)
        X_val = torch.tensor(X_train_n[n_tr:], dtype=torch.float32, device=device)
        Y_val = torch.tensor(Y_train_n[n_tr:], dtype=torch.float32, device=device)

        model = _PredictionMLP(input_dim, output_dim,
                               self.hidden_size, self.n_hidden).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr,
                                     weight_decay=self.weight_decay)
        criterion = nn.MSELoss()

        train_ds = TensorDataset(X_tr, Y_tr)
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)

        best_val_loss = float("inf")
        best_state = None
        wait = 0

        for epoch in range(self.max_epochs):
            model.train()
            for xb, yb in train_loader:
                mask = (torch.rand(xb.shape[0], 1, device=device) > self.p_mask).float()
                xb_masked = xb.clone()
                xb_masked[:, obs_dim:] = xb[:, obs_dim:] * mask

                optimizer.zero_grad()
                loss = criterion(model(xb_masked), yb)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                val_full = criterion(model(X_val), Y_val).item()
                X_val_m = X_val.clone()
                X_val_m[:, obs_dim:] = 0.0
                val_masked = criterion(model(X_val_m), Y_val).item()
                val_loss = 0.5 * val_full + 0.5 * val_masked

            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= self.patience:
                    break

        if best_state is not None:
            model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

        model.eval()
        norm_stats = {"X_mean": X_mean, "X_std": X_std, "Y_mean": Y_mean, "Y_std": Y_std}
        return model, norm_stats, device

    def _nn_evaluate(self, model, X_test, Y_test, norm_stats, device):
        """Evaluate model, returning MSE in original space."""
        X_n = (X_test - norm_stats["X_mean"]) / norm_stats["X_std"]
        X_t = torch.tensor(X_n, dtype=torch.float32, device=device)
        with torch.no_grad():
            pred_n = model(X_t).cpu().numpy()
        pred = pred_n * norm_stats["Y_std"] + norm_stats["Y_mean"]
        return float(np.mean((Y_test - pred) ** 2))

    def _nn_mvs(self, X_h_train, X_h_test, Y_train, Y_test, obs_dim):
        """Compute MVS using NN with group dropout, averaged over runs."""
        mvs_runs = []
        mse_m_runs = []
        mse_h_runs = []

        for run in range(self.n_runs):
            seed = run * 1000 + 42
            model, nstats, dev = self._train_nn(X_h_train, Y_train, obs_dim, seed)

            mse_full = self._nn_evaluate(model, X_h_test, Y_test, nstats, dev)

            X_masked = X_h_test.copy()
            X_masked[:, obs_dim:] = 0.0
            mse_masked = self._nn_evaluate(model, X_masked, Y_test, nstats, dev)

            mse_h_runs.append(mse_full)
            mse_m_runs.append(mse_masked)

            if mse_masked < 1e-15:
                mvs_runs.append(0.0)
            else:
                mvs_runs.append(float(np.clip(
                    (mse_masked - mse_full) / mse_masked, 0.0, 1.0
                )))

        avg_mse_m = float(np.mean(mse_m_runs))
        avg_mse_h = float(np.mean(mse_h_runs))

        if avg_mse_m < 1e-15:
            mvs = 0.0
        else:
            mvs = (avg_mse_m - avg_mse_h) / avg_mse_m
            mvs = float(np.clip(mvs, 0.0, 1.0))

        return mvs, avg_mse_m, avg_mse_h

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------

    def compute_mvs(self, observations):
        """
        Compute the prediction-based Markov Violation Score.

        Two-stage:
          1. Ridge regression (linear comparison, exact solution)
          2. NN with group dropout (nonlinear comparison, optional)
          Final MVS = max(Ridge MVS, NN MVS)

        Parameters
        ----------
        observations : np.ndarray, shape (T, N)

        Returns
        -------
        dict with keys:
            mvs : float — final MVS (max of ridge and nn), clipped to [0, 1]
            mvs_ridge : float — Ridge regression MVS
            mvs_nn : float — NN MVS (0.0 if use_nn=False)
            mse_markov : float — Markov model MSE (from best method)
            mse_history : float — history model MSE (from best method)
        """
        observations = np.asarray(observations, dtype=np.float32)
        T, N = observations.shape
        k = self.history_len

        X_markov, X_history, Y = self._build_datasets(observations, k)

        # 70/30 time-respecting split
        n_total = len(Y)
        n_train = int(0.7 * n_total)

        X_m_train, X_m_test = X_markov[:n_train], X_markov[n_train:]
        X_h_train, X_h_test = X_history[:n_train], X_history[n_train:]
        Y_train, Y_test = Y[:n_train], Y[n_train:]

        # Stage 1: Ridge
        mvs_ridge, mse_m_ridge, mse_h_ridge = self._ridge_mvs(
            X_m_train, X_m_test, X_h_train, X_h_test, Y_train, Y_test
        )

        # Stage 2: NN (optional)
        mvs_nn = 0.0
        mse_m_nn = mse_m_ridge
        mse_h_nn = mse_h_ridge
        if self.use_nn:
            mvs_nn, mse_m_nn, mse_h_nn = self._nn_mvs(
                X_h_train, X_h_test, Y_train, Y_test, obs_dim=N
            )

        # Final: take the max (captures both linear and nonlinear effects)
        if mvs_ridge >= mvs_nn:
            mvs = mvs_ridge
            mse_m = mse_m_ridge
            mse_h = mse_h_ridge
        else:
            mvs = mvs_nn
            mse_m = mse_m_nn
            mse_h = mse_h_nn

        return {
            "mvs": mvs,
            "mvs_ridge": mvs_ridge,
            "mvs_nn": mvs_nn,
            "mse_markov": mse_m,
            "mse_history": mse_h,
        }

    def run(self, observations, tau_max=5, alpha_level=0.05, var_names=None):
        """
        Interface-compatible entry point matching other CI backends.

        Calls compute_mvs() internally and constructs backward-compatible
        p_matrix / val_matrix outputs.

        Returns
        -------
        dict with keys:
            p_matrix : np.ndarray, shape (N, N, tau_max+1)
            val_matrix : np.ndarray, shape (N, N, tau_max+1)
            lag2_or_more_links : list
            prediction_mvs : float — the direct prediction-based MVS
            prediction_details : dict — full compute_mvs() output
        """
        if not isinstance(observations, np.ndarray):
            observations = np.asarray(observations)
        if observations.ndim != 2:
            raise ValueError("Observations array should be 2D: (time, variables).")

        T, N = observations.shape
        if var_names is None:
            var_names = [f"Var{i}" for i in range(N)]

        mvs_result = self.compute_mvs(observations)
        mvs = mvs_result["mvs"]

        # Backward-compatible matrices
        p_matrix = np.ones((N, N, tau_max + 1))
        val_matrix = np.zeros((N, N, tau_max + 1))

        if mvs > 0.01:
            for i in range(N):
                for lag in range(2, tau_max + 1):
                    p_matrix[i, i, lag] = 0.001
                    val_matrix[i, i, lag] = mvs

        lag2_or_more_links = []
        if mvs > 0.01:
            for i in range(N):
                for lag in range(2, tau_max + 1):
                    lag2_or_more_links.append((i, i, -lag, mvs))

        self.results = {
            "p_matrix": p_matrix,
            "val_matrix": val_matrix,
            "lag2_or_more_links": lag2_or_more_links,
            "prediction_mvs": mvs,
            "prediction_details": mvs_result,
        }

        logger.info(
            f"[PredictionMVS] T={T}, N={N}, MVS={mvs:.4f} "
            f"(ridge={mvs_result['mvs_ridge']:.4f}, "
            f"nn={mvs_result['mvs_nn']:.4f})"
        )
        return self.results
