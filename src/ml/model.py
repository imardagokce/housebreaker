"""Independent home- and away-goal regressors for football fixtures."""

from __future__ import annotations

import base64
import json
import pickle
from pathlib import Path
from typing import Any, Literal

import numpy as np
import polars as pl
from pydantic import BaseModel, ConfigDict, Field, ValidationError

Backend = Literal["xgboost", "lightgbm"]
DEFAULT_MODELS_DIRECTORY = Path(__file__).resolve().parents[2] / "models"
_TARGET_COLUMNS = ("home_goals", "away_goals")
_EXCLUDED_FEATURE_COLUMNS = {
    "fixture_date",
    "home_goals",
    "away_goals",
    "home_xg",
    "away_xg",
}


class ModelSettings(BaseModel):
    """Validated options shared by the supported gradient-boosting backends."""

    model_config = ConfigDict(frozen=True)

    backend: Backend = "xgboost"
    n_estimators: int = Field(default=200, ge=1)
    learning_rate: float = Field(default=0.05, gt=0, le=1, allow_inf_nan=False)
    max_depth: int = Field(default=4, ge=1)
    random_state: int = 42


class EvaluationMetrics(BaseModel):
    """Mean absolute error and root mean squared error for each goal target."""

    home_mae: float = Field(ge=0, allow_inf_nan=False)
    home_rmse: float = Field(ge=0, allow_inf_nan=False)
    away_mae: float = Field(ge=0, allow_inf_nan=False)
    away_rmse: float = Field(ge=0, allow_inf_nan=False)


class GoalRegressor:
    """Train and use independent XGBoost or LightGBM goal regressors."""

    def __init__(self, settings: ModelSettings | None = None) -> None:
        if settings is not None and not isinstance(settings, ModelSettings):
            raise TypeError("settings must be a ModelSettings instance.")
        self.settings = settings if settings is not None else ModelSettings()
        self.feature_columns: list[str] = []
        self.home_model: Any | None = None
        self.away_model: Any | None = None

    @property
    def is_fitted(self) -> bool:
        """Return whether both independent goal models have been trained."""
        return self.home_model is not None and self.away_model is not None

    def fit(
        self,
        training_data: pl.DataFrame,
        *,
        feature_columns: list[str] | None = None,
    ) -> "GoalRegressor":
        """Train home- and away-goal models from validated Polars data."""
        if not isinstance(training_data, pl.DataFrame):
            raise TypeError("training_data must be a Polars DataFrame.")
        if training_data.height < 2:
            raise ValueError("training_data must contain at least two fixtures.")
        missing_targets = sorted(set(_TARGET_COLUMNS) - set(training_data.columns))
        if missing_targets:
            raise ValueError(
                "training_data is missing target columns: " + ", ".join(missing_targets)
            )

        selected_features = (
            list(feature_columns)
            if feature_columns is not None
            else [
                column
                for column, dtype in training_data.schema.items()
                if column not in _EXCLUDED_FEATURE_COLUMNS and dtype.is_numeric()
            ]
        )
        if not selected_features:
            raise ValueError("At least one numeric model feature is required.")
        if len(set(selected_features)) != len(selected_features):
            raise ValueError("feature_columns must not contain duplicates.")
        missing_features = sorted(set(selected_features) - set(training_data.columns))
        if missing_features:
            raise ValueError(
                "training_data is missing feature columns: "
                + ", ".join(missing_features)
            )

        invalid_features = [
            column
            for column in selected_features
            if not training_data.schema[column].is_numeric()
        ]
        if invalid_features:
            raise TypeError(
                "Model features must be numeric: " + ", ".join(invalid_features)
            )
        selected = training_data.select(
            *[pl.col(column).cast(pl.Float64, strict=True) for column in selected_features],
            *[pl.col(column).cast(pl.Float64, strict=True) for column in _TARGET_COLUMNS],
        )
        if any(selected.null_count().row(0)):
            raise ValueError("Training features and goal targets must not contain nulls.")
        for column in selected.columns:
            if selected.filter(
                pl.col(column).is_nan()
                | pl.col(column).is_infinite()
                | (
                    (pl.col(column) < 0)
                    if column in _TARGET_COLUMNS
                    else pl.lit(False)
                )
            ).height:
                raise ValueError(
                    f"{column} must contain finite values"
                    + (" and non-negative goals." if column in _TARGET_COLUMNS else ".")
                )

        try:
            feature_values = selected.select(selected_features).to_numpy()
            home_targets = selected.get_column("home_goals").to_numpy()
            away_targets = selected.get_column("away_goals").to_numpy()
            home_model = self._new_estimator()
            away_model = self._new_estimator()
            home_model.fit(feature_values, home_targets)
            away_model.fit(feature_values, away_targets)
        except (ImportError, ValueError, RuntimeError) as exc:
            raise RuntimeError(
                f"Unable to train the {self.settings.backend} goal models: {exc}"
            ) from exc

        self.home_model = home_model
        self.away_model = away_model
        self.feature_columns = list(selected_features)
        return self

    def predict(self, features: pl.DataFrame) -> pl.DataFrame:
        """Predict non-negative expected home and away goals."""
        feature_values = self._feature_values(features)
        home_predictions = np.maximum(self._predict_model(self.home_model, feature_values), 0.0)
        away_predictions = np.maximum(self._predict_model(self.away_model, feature_values), 0.0)
        prediction_data: dict[str, Any] = {
            "home_xg": home_predictions,
            "away_xg": away_predictions,
        }
        for identifier in ("fixture_date", "home_team", "away_team"):
            if identifier in features.columns:
                prediction_data[identifier] = features.get_column(identifier)
        return pl.DataFrame(prediction_data)

    def evaluate(self, test_data: pl.DataFrame) -> EvaluationMetrics:
        """Evaluate both models on held-out fixtures using MAE and RMSE."""
        if not isinstance(test_data, pl.DataFrame):
            raise TypeError("test_data must be a Polars DataFrame.")
        missing_targets = sorted(set(_TARGET_COLUMNS) - set(test_data.columns))
        if missing_targets:
            raise ValueError(
                "test_data is missing target columns: " + ", ".join(missing_targets)
            )
        predictions = self.predict(test_data)
        targets = test_data.select(
            *[pl.col(column).cast(pl.Float64, strict=True) for column in _TARGET_COLUMNS]
        )
        if any(targets.null_count().row(0)):
            raise ValueError("Test goal targets must not contain nulls.")
        metrics: dict[str, float] = {}
        for side in ("home", "away"):
            actual = targets.get_column(f"{side}_goals").to_numpy()
            predicted = predictions.get_column(f"{side}_xg").to_numpy()
            if not np.isfinite(actual).all() or (actual < 0).any():
                raise ValueError(f"{side}_goals must contain finite, non-negative values.")
            error = predicted - actual
            metrics[f"{side}_mae"] = float(np.mean(np.abs(error)))
            metrics[f"{side}_rmse"] = float(np.sqrt(np.mean(np.square(error))))
        try:
            return EvaluationMetrics.model_validate(metrics)
        except ValidationError as exc:
            raise ValueError(f"Model evaluation metrics failed validation: {exc}") from exc

    def save(self, path: str | Path | None = None) -> Path:
        """Save the fitted estimators as a pickle or native XGBoost JSON file."""
        if not self.is_fitted:
            raise RuntimeError("Train both goal models before saving.")
        output_path = Path(path) if path is not None else DEFAULT_MODELS_DIRECTORY / "goal_regressor.pkl"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        suffix = output_path.suffix.casefold()
        if suffix == ".pkl":
            try:
                with output_path.open("wb") as model_file:
                    pickle.dump(self, model_file, protocol=pickle.HIGHEST_PROTOCOL)
            except (OSError, pickle.PickleError, TypeError) as exc:
                raise OSError(f"Unable to save model artefact to {output_path}.") from exc
            return output_path
        if suffix == ".json":
            self._save_xgboost_json(output_path)
            return output_path
        raise ValueError("Model artefact paths must end in .pkl or .json.")

    @classmethod
    def load(cls, path: str | Path) -> "GoalRegressor":
        """Load a pickle or native XGBoost JSON model artefact."""
        model_path = Path(path)
        suffix = model_path.suffix.casefold()
        if suffix == ".pkl":
            try:
                with model_path.open("rb") as model_file:
                    model = pickle.load(model_file)
            except (
                OSError,
                pickle.PickleError,
                EOFError,
                ImportError,
                AttributeError,
            ) as exc:
                raise OSError(f"Unable to load model artefact from {model_path}.") from exc
            if not isinstance(model, cls) or not model.is_fitted:
                raise ValueError(f"The pickle at {model_path} is not a fitted GoalRegressor.")
            return model
        if suffix == ".json":
            return cls._load_xgboost_json(model_path)
        raise ValueError("Model artefact paths must end in .pkl or .json.")

    def _new_estimator(self) -> Any:
        if self.settings.backend == "xgboost":
            try:
                from xgboost import XGBRegressor
            except ImportError as exc:
                raise ImportError("Install xgboost to use the XGBoost backend.") from exc
            return XGBRegressor(
                objective="count:poisson",
                eval_metric="poisson-nloglik",
                n_estimators=self.settings.n_estimators,
                learning_rate=self.settings.learning_rate,
                max_depth=self.settings.max_depth,
                random_state=self.settings.random_state,
                n_jobs=1,
                verbosity=0,
            )
        try:
            from lightgbm import LGBMRegressor
        except ImportError as exc:
            raise ImportError("Install lightgbm to use the LightGBM backend.") from exc
        return LGBMRegressor(
            objective="poisson",
            n_estimators=self.settings.n_estimators,
            learning_rate=self.settings.learning_rate,
            max_depth=self.settings.max_depth,
            random_state=self.settings.random_state,
            n_jobs=1,
            verbosity=-1,
        )

    def _feature_values(self, features: pl.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Train or load the goal models before prediction.")
        if not isinstance(features, pl.DataFrame):
            raise TypeError("features must be a Polars DataFrame.")
        if features.is_empty():
            raise ValueError("features must contain at least one fixture.")
        missing = sorted(set(self.feature_columns) - set(features.columns))
        if missing:
            raise ValueError(
                "features is missing model columns: " + ", ".join(missing)
            )
        invalid = [
            column
            for column in self.feature_columns
            if not features.schema[column].is_numeric()
        ]
        if invalid:
            raise TypeError("Model features must be numeric: " + ", ".join(invalid))
        values = features.select(
            *[pl.col(column).cast(pl.Float64, strict=True) for column in self.feature_columns]
        )
        if any(values.null_count().row(0)):
            raise ValueError("Model features must not contain null values.")
        array = values.to_numpy()
        if not np.isfinite(array).all():
            raise ValueError("Model features must contain only finite values.")
        return array

    def _predict_model(self, model: Any | None, values: np.ndarray) -> np.ndarray:
        if model is None:
            raise RuntimeError("Train or load both goal models before prediction.")
        try:
            from xgboost import Booster, DMatrix

            if isinstance(model, Booster):
                predictions = model.predict(DMatrix(values))
            else:
                predictions = model.predict(values)
        except ImportError:
            predictions = model.predict(values)
        result = np.asarray(predictions, dtype=np.float64)
        if result.ndim != 1 or not np.isfinite(result).all():
            raise RuntimeError("The goal model returned invalid predictions.")
        return result

    def _save_xgboost_json(self, path: Path) -> None:
        if self.settings.backend != "xgboost":
            raise ValueError("Native .json artefacts are supported only by XGBoost.")
        try:
            import xgboost
            from xgboost import XGBRegressor

            if not isinstance(self.home_model, XGBRegressor) or not isinstance(
                self.away_model, XGBRegressor
            ):
                raise ValueError("Native .json export requires fitted XGBoost regressors.")
            payload = {
                "format_version": 1,
                "backend": "xgboost",
                "xgboost_version": xgboost.__version__,
                "settings": self.settings.model_dump(),
                "feature_columns": self.feature_columns,
                "home_model": base64.b64encode(
                    bytes(self.home_model.get_booster().save_raw(raw_format="json"))
                ).decode("ascii"),
                "away_model": base64.b64encode(
                    bytes(self.away_model.get_booster().save_raw(raw_format="json"))
                ).decode("ascii"),
            }
            path.write_text(json.dumps(payload), encoding="utf-8")
        except (ImportError, OSError, ValueError, RuntimeError) as exc:
            raise OSError(f"Unable to save XGBoost JSON artefact to {path}.") from exc

    @classmethod
    def _load_xgboost_json(cls, path: Path) -> "GoalRegressor":
        try:
            import xgboost
            from xgboost import Booster

            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("format_version") != 1 or payload.get("backend") != "xgboost":
                raise ValueError("Unsupported model artefact format.")
            settings = ModelSettings.model_validate(payload["settings"])
            if settings.backend != "xgboost":
                raise ValueError("JSON model artefact backend does not match its settings.")
            feature_columns = payload["feature_columns"]
            if (
                not isinstance(feature_columns, list)
                or not feature_columns
                or not all(isinstance(column, str) for column in feature_columns)
            ):
                raise ValueError("The model artefact has invalid feature columns.")
            home_model = Booster()
            away_model = Booster()
            home_model.load_model(
                bytearray(base64.b64decode(payload["home_model"], validate=True))
            )
            away_model.load_model(
                bytearray(base64.b64decode(payload["away_model"], validate=True))
            )
            model = cls(settings)
            model.feature_columns = feature_columns
            model.home_model = home_model
            model.away_model = away_model
            _ = xgboost.__version__
            return model
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise OSError(f"Unable to load XGBoost JSON artefact from {path}.") from exc
