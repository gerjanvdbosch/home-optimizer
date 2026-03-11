import logging

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

logger = logging.getLogger(__name__)


# =========================================================
# MODEL SELECTOR
# =========================================================
class ModelSelector:
    """
    Selecteert automatisch het beste regressiemodel via cross-validation.
    Herbruikbaar voor PerfMap, Solar, Load, en andere predictors.

    Selectiecriterium: R2_mean - 0.5 * R2_std
    Dit bestraft instabiele modellen (hoge variantie over folds).

    Gebruik:
        model, name, score = ModelSelector.select(X, y, "UFH P_el")
    """

    @staticmethod
    def _candidates() -> dict:
        """Geeft altijd verse (ongefitte) model-instanties terug."""
        return {
            "LinearRegression": LinearRegression(),
            "Ridge": Ridge(alpha=1.0),
            "Poly2": make_pipeline(
                PolynomialFeatures(2), Ridge(alpha=1.0)
            ),
            "RandomForest": RandomForestRegressor(
                n_estimators=150, max_depth=5,
                min_samples_leaf=8, random_state=42,
            ),
            "GradientBoosting": GradientBoostingRegressor(
                n_estimators=200, max_depth=4,
                learning_rate=0.05, min_samples_leaf=8,
                random_state=42,
            ),
            "HistGradientBoosting": HistGradientBoostingRegressor(
                max_iter=200, max_depth=4,
                learning_rate=0.05, min_samples_leaf=8,
                random_state=42,
            ),
        }

    @staticmethod
    def select(
        X,
        y,
        label:      str,
        cv:         int   = 5,
        min_r2:     float = 0.0,
        variance_penalty: float = 0.5,
    ):
        """
        Selecteer het beste model op basis van cross-validation.

        Parameters
        ----------
        X               : features (DataFrame of array)
        y               : target (Series of array)
        label           : naam voor logging (bijv. "UFH P_el")
        cv              : aantal cross-validation folds
        min_r2          : minimale adjusted R2 om model te accepteren.
                          Bij geen enkel model boven min_r2 → None teruggeven.
        variance_penalty: gewicht waarmee std van R2 wordt afgestraft.
                          adjusted_r2 = mean - penalty * std

        Returns
        -------
        (model, name, adjusted_score) of (None, None, None) bij mislukking
        """
        if len(X) < cv * 2:
            logger.warning(
                f"[ModelSelector] {label}: te weinig data ({len(X)} rijen) "
                f"voor {cv}-fold CV — sla over."
            )
            return None, None, None

        best_name     = None
        best_adjusted = -999.0
        best_mean     = None
        best_std      = None

        candidates = ModelSelector._candidates()

        for name, model in candidates.items():
            try:
                scores = cross_val_score(model, X, y, cv=cv, scoring="r2")
                mean   = float(scores.mean())
                std    = float(scores.std())
            except Exception as e:
                logger.warning(f"[ModelSelector] {label} {name}: mislukt ({e})")
                continue

            adjusted = mean - variance_penalty * std

            logger.info(
                f"[ModelSelector] {label} {name}: "
                f"R2={mean:.3f}+-{std:.3f}  adjusted={adjusted:.3f}"
            )

            if adjusted > best_adjusted:
                best_adjusted = adjusted
                best_mean     = mean
                best_std      = std
                best_name     = name

        if best_name is None or best_adjusted < min_r2:
            logger.warning(
                f"[ModelSelector] {label}: geen enkel model haalt min_r2={min_r2:.2f} "
                f"(beste={best_adjusted:.3f}) — geen model opgeslagen."
            )
            return None, None, None

        # Fit het winnende model op alle data (verse instantie)
        winner = ModelSelector._candidates()[best_name]
        winner.fit(X, y)

        logger.info(
            f"[ModelSelector] {label} ✓ winnaar: {best_name}  "
            f"R2={best_mean:.3f}+-{best_std:.3f}  adjusted={best_adjusted:.3f}"
        )

        return winner, best_name, best_adjusted