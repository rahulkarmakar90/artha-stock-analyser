"""
ml_analysis.py — ML-based stock prediction and walk-forward backtesting.

Models:
  - Linear Regression       : predict next-day return %
  - Random Forest           : classify direction (up/down/neutral) + feature importances
  - Gradient Boosting       : same as RF, used in ensemble
  - Gaussian Naive Bayes    : Bayesian probabilistic direction prediction
  - Ensemble                : average of RF + GB probabilities

WalkForwardBacktester validates predictions against 5 years of history.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB

FEATURE_COLS = [
    "RSI", "MACD", "MACD_Signal", "MACD_histogram",
    "BB_position", "SMA_ratio", "volume_ratio",
    "return_1d", "return_5d", "return_20d", "volatility",
]


class MLAnalyzer:

    # ── Feature Engineering ────────────────────────────────────────────────

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all 11 ML features from OHLCV data."""
        d = df.copy()

        # RSI (14)
        delta = d["Close"].diff()
        gain  = delta.where(delta > 0, 0.0).rolling(14).mean()
        loss  = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
        rs    = gain / loss.replace(0, np.nan)
        d["RSI"] = 100 - (100 / (1 + rs))

        # MACD
        ema12 = d["Close"].ewm(span=12).mean()
        ema26 = d["Close"].ewm(span=26).mean()
        d["MACD"]           = ema12 - ema26
        d["MACD_Signal"]    = d["MACD"].ewm(span=9).mean()
        d["MACD_histogram"] = d["MACD"] - d["MACD_Signal"]

        # Bollinger Band position
        sma20   = d["Close"].rolling(20).mean()
        std20   = d["Close"].rolling(20).std()
        bb_up   = sma20 + 2 * std20
        bb_lo   = sma20 - 2 * std20
        bb_span = (bb_up - bb_lo).replace(0, np.nan)
        d["BB_position"] = ((d["Close"] - bb_lo) / bb_span).clip(0, 1)

        # SMA ratio
        sma5            = d["Close"].rolling(5).mean()
        d["SMA_ratio"]  = (sma5 / sma20.replace(0, np.nan)).fillna(1.0)

        # Volume ratio
        vol_ma           = d["Volume"].rolling(20).mean().replace(0, np.nan)
        d["volume_ratio"] = (d["Volume"] / vol_ma).fillna(1.0)

        # Momentum returns
        d["return_1d"]  = d["Close"].pct_change(1)
        d["return_5d"]  = d["Close"].pct_change(5)
        d["return_20d"] = d["Close"].pct_change(20)

        # Volatility
        d["volatility"] = d["return_1d"].rolling(20).std()

        return d

    def build_target(self, df: pd.DataFrame) -> pd.Series:
        """Next-day direction: +1 (>+0.5%), -1 (<-0.5%), 0 otherwise."""
        next_ret = df["Close"].shift(-1) / df["Close"] - 1
        target = pd.Series(0, index=df.index, name="target")
        target[next_ret >  0.005] =  1
        target[next_ret < -0.005] = -1
        return target

    def prepare_Xy(self, df: pd.DataFrame):
        """Return (X_array, y_array, feature_names) ready for sklearn."""
        feat = self.engineer_features(df)
        tgt  = self.build_target(feat)

        combined = feat[FEATURE_COLS].copy()
        combined["target"] = tgt
        combined = combined.dropna()
        # Drop last row — target is NaN (no next-day close)
        combined = combined.iloc[:-1]

        X = combined[FEATURE_COLS].values
        y = combined["target"].values.astype(int)
        return X, y, FEATURE_COLS

    # ── Linear Regression ─────────────────────────────────────────────────

    def run_linear_regression(self, df: pd.DataFrame) -> dict:
        """Predict next-day return % using LinearRegression."""
        feat      = self.engineer_features(df)
        next_ret  = df["Close"].shift(-1) / df["Close"] - 1

        combined  = feat[FEATURE_COLS].copy()
        combined["next_ret"] = next_ret
        combined  = combined.dropna().iloc[:-1]

        if len(combined) < 50:
            return {"error": "Insufficient data for regression"}

        X = combined[FEATURE_COLS].values
        y = combined["next_ret"].values

        scaler = StandardScaler()
        X_sc   = scaler.fit_transform(X)

        model = LinearRegression()
        model.fit(X_sc, y)

        r2 = float(model.score(X_sc, y))

        # Predict on the latest row
        latest_feat = feat[FEATURE_COLS].dropna().iloc[[-1]]
        X_latest    = scaler.transform(latest_feat.values)
        pred_ret    = float(model.predict(X_latest)[0])
        curr_price  = float(df["Close"].iloc[-1])

        return {
            "predicted_return_pct": round(pred_ret, 6),
            "predicted_price":      round(curr_price * (1 + pred_ret), 2),
            "model_r2_train":       round(r2, 4),
            "current_price":        round(curr_price, 2),
        }

    # ── Random Forest ─────────────────────────────────────────────────────

    def run_random_forest(self, df: pd.DataFrame) -> dict:
        X, y, feature_names = self.prepare_Xy(df)
        if len(X) < 50:
            return {"error": "Insufficient data"}

        clf = RandomForestClassifier(
            n_estimators=100, max_depth=6, random_state=42,
            class_weight="balanced", n_jobs=-1
        )
        clf.fit(X, y)

        latest_X = self._latest_feature_row(df)
        if latest_X is None:
            return {"error": "Could not extract latest features"}

        direction  = int(clf.predict(latest_X)[0])
        proba      = clf.predict_proba(latest_X)[0]
        class_map  = {c: i for i, c in enumerate(clf.classes_)}
        p_down     = float(proba[class_map.get(-1, 0)]) if -1 in class_map else 0.0
        p_neutral  = float(proba[class_map.get( 0, 0)]) if  0 in class_map else 0.0
        p_up       = float(proba[class_map.get( 1, 0)]) if  1 in class_map else 0.0

        importances = {
            name: round(float(imp), 4)
            for name, imp in zip(feature_names, clf.feature_importances_)
        }

        return {
            "direction":          direction,
            "direction_label":    self._dir_label(direction),
            "confidence":         round(max(p_up, p_down, p_neutral) * 100, 2),
            "prob_up":            round(p_up, 4),
            "prob_down":          round(p_down, 4),
            "prob_neutral":       round(p_neutral, 4),
            "feature_importances": importances,
        }

    # ── Gradient Boosting ─────────────────────────────────────────────────

    def run_gradient_boosting(self, df: pd.DataFrame) -> dict:
        X, y, feature_names = self.prepare_Xy(df)
        if len(X) < 50:
            return {"error": "Insufficient data"}

        clf = GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42
        )
        clf.fit(X, y)

        latest_X = self._latest_feature_row(df)
        if latest_X is None:
            return {"error": "Could not extract latest features"}

        direction  = int(clf.predict(latest_X)[0])
        proba      = clf.predict_proba(latest_X)[0]
        class_map  = {c: i for i, c in enumerate(clf.classes_)}
        p_down     = float(proba[class_map.get(-1, 0)]) if -1 in class_map else 0.0
        p_neutral  = float(proba[class_map.get( 0, 0)]) if  0 in class_map else 0.0
        p_up       = float(proba[class_map.get( 1, 0)]) if  1 in class_map else 0.0

        importances = {
            name: round(float(imp), 4)
            for name, imp in zip(feature_names, clf.feature_importances_)
        }

        return {
            "direction":           direction,
            "direction_label":     self._dir_label(direction),
            "confidence":          round(max(p_up, p_down, p_neutral) * 100, 2),
            "prob_up":             round(p_up, 4),
            "prob_down":           round(p_down, 4),
            "prob_neutral":        round(p_neutral, 4),
            "feature_importances": importances,
        }

    # ── Gaussian Naive Bayes (Bayesian) ───────────────────────────────────

    def run_gaussian_naive_bayes(self, df: pd.DataFrame) -> dict:
        X, y, feature_names = self.prepare_Xy(df)
        if len(X) < 50:
            return {"error": "Insufficient data"}

        gnb = GaussianNB()
        gnb.fit(X, y)

        latest_X = self._latest_feature_row(df)
        if latest_X is None:
            return {"error": "Could not extract latest features"}

        direction  = int(gnb.predict(latest_X)[0])
        proba      = gnb.predict_proba(latest_X)[0]
        class_map  = {c: i for i, c in enumerate(gnb.classes_)}
        p_down     = float(proba[class_map.get(-1, 0)]) if -1 in class_map else 0.0
        p_neutral  = float(proba[class_map.get( 0, 0)]) if  0 in class_map else 0.0
        p_up       = float(proba[class_map.get( 1, 0)]) if  1 in class_map else 0.0

        # Per-indicator Bayesian likelihoods using stored Gaussian parameters
        x_obs = latest_X[0]   # shape (n_features,)
        per_indicator = {}
        for fi, fname in enumerate(feature_names):
            likelihoods = {}
            for ci, cls in enumerate(gnb.classes_):
                mu  = float(gnb.theta_[ci, fi])
                var = float(gnb.var_[ci, fi])
                std = max(np.sqrt(var), 1e-9)
                # Gaussian PDF at observed value
                lk = (1.0 / (std * np.sqrt(2 * np.pi))) * \
                     np.exp(-0.5 * ((x_obs[fi] - mu) / std) ** 2)
                label = {-1: "p_down", 0: "p_neutral", 1: "p_up"}.get(cls, str(cls))
                likelihoods[label] = round(float(lk), 6)
            per_indicator[fname] = likelihoods

        return {
            "direction":               direction,
            "direction_label":         self._dir_label(direction),
            "prob_up":                 round(p_up, 4),
            "prob_down":               round(p_down, 4),
            "prob_neutral":            round(p_neutral, 4),
            "per_indicator_likelihood": per_indicator,
        }

    # ── Ensemble ──────────────────────────────────────────────────────────

    def run_ensemble(self, rf: dict, gb: dict) -> dict:
        if "error" in rf or "error" in gb:
            return {"error": "One or more models failed"}
        p_up      = (rf["prob_up"]      + gb["prob_up"])      / 2
        p_down    = (rf["prob_down"]    + gb["prob_down"])    / 2
        p_neutral = (rf["prob_neutral"] + gb["prob_neutral"]) / 2
        probs     = {1: p_up, -1: p_down, 0: p_neutral}
        direction = max(probs, key=probs.get)
        return {
            "direction":      direction,
            "direction_label": self._dir_label(direction),
            "confidence":     round(max(p_up, p_down, p_neutral) * 100, 2),
            "prob_up":        round(p_up, 4),
            "prob_down":      round(p_down, 4),
            "prob_neutral":   round(p_neutral, 4),
        }

    # ── Run All ───────────────────────────────────────────────────────────

    def run_all(self, df_5y: pd.DataFrame) -> dict:
        if df_5y is None or len(df_5y) < 300:
            return {"error": "Insufficient history (need ~5 years of daily data)"}
        try:
            lr  = self.run_linear_regression(df_5y)
            rf  = self.run_random_forest(df_5y)
            gb  = self.run_gradient_boosting(df_5y)
            gnb = self.run_gaussian_naive_bayes(df_5y)
            ens = self.run_ensemble(rf, gb)
            return {
                "linear_regression":  lr,
                "random_forest":      rf,
                "gradient_boosting":  gb,
                "bayesian":           gnb,
                "ensemble":           ens,
            }
        except Exception as e:
            return {"error": str(e)}

    # ── Helpers ───────────────────────────────────────────────────────────

    def _latest_feature_row(self, df: pd.DataFrame):
        """Return the latest complete feature row as a 2D array for sklearn."""
        try:
            feat = self.engineer_features(df)
            row  = feat[FEATURE_COLS].dropna().iloc[[-1]]
            return row.values
        except Exception:
            return None

    @staticmethod
    def _dir_label(d: int) -> str:
        return {1: "up", -1: "down", 0: "neutral"}.get(d, "neutral")


# ── Walk-Forward Backtester ───────────────────────────────────────────────

class WalkForwardBacktester:

    def run(self, df_5y: pd.DataFrame, analyzer: MLAnalyzer) -> dict:
        """
        Walk-forward validation using a rolling 252-day training window.
        For each test day:
          - Train RF on previous 252 days
          - Predict direction for that day
          - Compare against actual direction
        Returns accuracy, range accuracy, cumulative returns, feature importances.
        """
        X, y, feature_names = analyzer.prepare_Xy(df_5y)
        n = len(X)
        train_window = 252

        if n < train_window + 30:
            return {"error": f"Need at least {train_window + 30} rows, got {n}"}

        # Pre-compute actual daily returns for strategy simulation
        feat_df      = analyzer.engineer_features(df_5y)
        tgt_series   = analyzer.build_target(feat_df)
        combined     = feat_df[FEATURE_COLS].copy()
        combined["target"]    = tgt_series
        combined["ret_1d"]    = feat_df["Close"].pct_change()
        combined["close"]     = feat_df["Close"]
        combined["volatility"] = feat_df["volatility"]
        combined = combined.dropna().iloc[:-1]   # drop last (no next-day close)

        X_all   = combined[FEATURE_COLS].values
        y_all   = combined["target"].values.astype(int)
        rets    = combined["ret_1d"].values
        closes  = combined["close"].values
        vols    = combined["volatility"].values

        correct_direction = []
        in_range          = []
        strategy_rets     = []
        bnh_rets          = []
        n_trades          = 0
        last_rf           = None

        # Linear regression for range prediction (refit every 20 steps for speed)
        scaler_lr = StandardScaler()
        lr_model  = LinearRegression()
        lr_fitted = False

        for t in range(train_window, n - 1):
            X_train = X_all[t - train_window: t]
            y_train = y_all[t - train_window: t]
            X_test  = X_all[t: t + 1]
            y_test  = y_all[t]
            actual_ret = rets[t]

            # Train RF
            rf = RandomForestClassifier(
                n_estimators=50, max_depth=5, random_state=42,
                class_weight="balanced", n_jobs=-1
            )
            rf.fit(X_train, y_train)
            last_rf = rf

            y_pred = int(rf.predict(X_test)[0])

            # Direction accuracy
            correct_direction.append(int(y_pred == y_test))

            # Range accuracy (±1.96σ from current close)
            curr_vol = float(vols[t]) if not np.isnan(vols[t]) else 0.01
            curr_close = float(closes[t])
            pred_low  = curr_close * (1 - 1.96 * curr_vol)
            pred_high = curr_close * (1 + 1.96 * curr_vol)
            actual_next_close = curr_close * (1 + actual_ret)
            in_range.append(int(pred_low <= actual_next_close <= pred_high))

            # Strategy return: invest if predicted up, else hold cash
            if y_pred == 1:
                strategy_rets.append(actual_ret)
                n_trades += 1
            else:
                strategy_rets.append(0.0)

            bnh_rets.append(actual_ret)

        if not correct_direction:
            return {"error": "No test predictions generated"}

        # Metrics
        dir_acc    = float(np.mean(correct_direction)) * 100
        range_acc  = float(np.mean(in_range)) * 100
        strat_cum  = (float(np.prod([1 + r for r in strategy_rets])) - 1) * 100
        bnh_cum    = (float(np.prod([1 + r for r in bnh_rets])) - 1) * 100
        alpha      = strat_cum - bnh_cum

        # Feature importances from last RF fit
        fi = {}
        if last_rf is not None:
            fi = {
                name: round(float(imp), 4)
                for name, imp in zip(feature_names, last_rf.feature_importances_)
            }

        return {
            "direction_accuracy_pct":          round(dir_acc, 2),
            "range_accuracy_pct":              round(range_acc, 2),
            "cumulative_strategy_return_pct":  round(strat_cum, 2),
            "cumulative_bnh_return_pct":       round(bnh_cum, 2),
            "alpha_pct":                       round(alpha, 2),
            "n_test_days":                     len(correct_direction),
            "n_trades":                        n_trades,
            "feature_importances":             fi,
        }
