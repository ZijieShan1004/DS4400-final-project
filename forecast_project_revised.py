
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

DATA_PATH = "time_series_covid19_vaccine_doses_admin_global.csv"
OUTPUT_DIR = "outputs"

LAG_DAYS = 7
HORIZON = 7
TEST_RATIO = 0.20

# "cumulative" keeps your original idea.
# "increment" predicts daily new doses, which is usually harder but more realistic.
TARGET_MODE = "cumulative"


def load_dataset(path: str) -> pd.DataFrame:
    """Load the wide vaccination file and convert it to country-date format."""
    df = pd.read_csv(path)

    id_cols = [
        "UID", "iso2", "iso3", "code3", "FIPS", "Admin2", "Province_State",
        "Country_Region", "Lat", "Long_", "Combined_Key", "Population"
    ]
    date_cols = [col for col in df.columns if col not in id_cols]

    long_df = df.melt(
        id_vars=["Country_Region"],
        value_vars=date_cols,
        var_name="Date",
        value_name="Doses_admin"
    )

    long_df["Date"] = pd.to_datetime(long_df["Date"], errors="coerce")
    long_df["Doses_admin"] = pd.to_numeric(long_df["Doses_admin"], errors="coerce")
    long_df = long_df.dropna(subset=["Country_Region", "Date", "Doses_admin"])

    grouped = (
        long_df.groupby(["Country_Region", "Date"], as_index=False)["Doses_admin"]
        .sum()
        .sort_values(["Country_Region", "Date"])
        .reset_index(drop=True)
    )

    return grouped


def create_features(df: pd.DataFrame, lag_days: int = 7, horizon: int = 7,
                    target_mode: str = "cumulative") -> pd.DataFrame:
    """Create lag, rolling, calendar, and target features."""
    frames = []

    for country, g in tqdm(df.groupby("Country_Region"), desc="Feature Engineering"):
        g = g.copy().sort_values("Date").reset_index(drop=True)

        if len(g) < lag_days + horizon + 1:
            continue

        g["time_index"] = np.arange(len(g))
        g["day_of_week"] = g["Date"].dt.dayofweek
        g["day_of_month"] = g["Date"].dt.day
        g["month"] = g["Date"].dt.month
        g["day_of_year"] = g["Date"].dt.dayofyear
        g["year"] = g["Date"].dt.year

        g["log_doses"] = np.log1p(g["Doses_admin"])
        g["daily_increment"] = g["Doses_admin"].diff().clip(lower=0)
        g["growth_rate"] = g["daily_increment"] / (g["Doses_admin"].shift(1) + 1)

        base_series = g["Doses_admin"] if target_mode == "cumulative" else g["daily_increment"]

        for i in range(1, lag_days + 1):
            g[f"lag_{i}"] = base_series.shift(i)

        g["rolling_mean_7"] = base_series.shift(1).rolling(7).mean()
        g["rolling_std_7"] = base_series.shift(1).rolling(7).std()
        g["rolling_min_7"] = base_series.shift(1).rolling(7).min()
        g["rolling_max_7"] = base_series.shift(1).rolling(7).max()

        for step in range(1, horizon + 1):
            g[f"target_{step}"] = base_series.shift(-step)

        frames.append(g)

    data = pd.concat(frames, ignore_index=True)
    data = data.dropna().reset_index(drop=True)
    return data


def split_data(data: pd.DataFrame, test_ratio: float = 0.20):
    """Chronological train/test split using dates only."""
    dates = np.array(sorted(data["Date"].unique()))
    split_idx = int(len(dates) * (1 - test_ratio))
    split_idx = min(max(split_idx, 1), len(dates) - 1)
    cutoff = dates[split_idx]

    train = data[data["Date"] < cutoff].copy()
    test = data[data["Date"] >= cutoff].copy()

    return train, test, pd.Timestamp(cutoff)


def get_columns():
    target_cols = [f"target_{i}" for i in range(1, HORIZON + 1)]

    numeric_cols = [
        "time_index", "day_of_week", "day_of_month", "month", "day_of_year", "year",
        "Doses_admin", "log_doses", "daily_increment", "growth_rate",
        "rolling_mean_7", "rolling_std_7", "rolling_min_7", "rolling_max_7"
    ]
    numeric_cols += [f"lag_{i}" for i in range(1, LAG_DAYS + 1)]

    categorical_cols = ["Country_Region"]
    feature_cols = numeric_cols + categorical_cols

    return feature_cols, target_cols, numeric_cols, categorical_cols


def build_preprocessor(numeric_cols, categorical_cols):
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )


def mean_absolute_percentage_error_safe(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = np.maximum(np.abs(y_true), eps)
    return np.mean(np.abs((y_true - y_pred) / denom)) * 100


def symmetric_mape(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = np.maximum(np.abs(y_true) + np.abs(y_pred), eps)
    return np.mean(2 * np.abs(y_pred - y_true) / denom) * 100


def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error_safe(y_true, y_pred)
    smape = symmetric_mape(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape,
        "sMAPE": smape,
        "R2": r2
    }


def evaluate_by_horizon(y_true, y_pred):
    rows = []
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    for i in range(y_true.shape[1]):
        metrics = evaluate_model(y_true[:, i], y_pred[:, i])
        metrics["horizon_day"] = i + 1
        rows.append(metrics)

    return pd.DataFrame(rows)


def build_models(preprocessor):
    models = {
        "LinearRegression": Pipeline([
            ("prep", preprocessor),
            ("model", LinearRegression())
        ]),
        "RandomForest": Pipeline([
            ("prep", preprocessor),
            ("model", RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            ))
        ]),
        "GradientBoosting": Pipeline([
            ("prep", preprocessor),
            ("model", MultiOutputRegressor(
                GradientBoostingRegressor(
                    n_estimators=100,
                    learning_rate=0.05,
                    max_depth=3,
                    random_state=42
                )
            ))
        ]),
        "NeuralNetwork": Pipeline([
            ("prep", preprocessor),
            ("model", MLPRegressor(
                hidden_layer_sizes=(128, 64),
                activation="relu",
                solver="adam",
                learning_rate_init=0.001,
                max_iter=300,
                early_stopping=True,
                random_state=42
            ))
        ]),
    }
    return models


def naive_baseline_predictions(X_test: pd.DataFrame) -> np.ndarray:
    """Simple baseline: repeat lag_1 across all forecast steps."""
    last_value = X_test["lag_1"].to_numpy().reshape(-1, 1)
    return np.repeat(last_value, HORIZON, axis=1)


def save_dataset_summary(raw_df: pd.DataFrame, feature_df: pd.DataFrame, cutoff_date):
    Path(OUTPUT_DIR).mkdir(exist_ok=True)

    summary = pd.DataFrame([{
        "raw_rows": len(raw_df),
        "feature_rows": len(feature_df),
        "num_countries": raw_df["Country_Region"].nunique(),
        "start_date": raw_df["Date"].min(),
        "end_date": raw_df["Date"].max(),
        "target_mode": TARGET_MODE,
        "lag_days": LAG_DAYS,
        "horizon": HORIZON,
        "train_test_cutoff": cutoff_date
    }])
    summary.to_csv(Path(OUTPUT_DIR) / "dataset_summary.csv", index=False)

    missing = feature_df.isna().sum().sort_values(ascending=False).reset_index()
    missing.columns = ["column", "missing_count"]
    missing["missing_pct"] = missing["missing_count"] / len(feature_df) * 100
    missing.to_csv(Path(OUTPUT_DIR) / "missing_values_summary.csv", index=False)


def save_predictions(test_df: pd.DataFrame, y_true: pd.DataFrame, model_predictions: dict):
    rows = test_df[["Country_Region", "Date"]].copy().reset_index(drop=True)

    for i in range(HORIZON):
        rows[f"actual_day_{i+1}"] = y_true.iloc[:, i].values

    for model_name, preds in model_predictions.items():
        for i in range(HORIZON):
            rows[f"{model_name}_pred_day_{i+1}"] = preds[:, i]

    rows.to_csv(Path(OUTPUT_DIR) / "predictions.csv", index=False)


def save_metrics(results_rows, horizon_frames):
    Path(OUTPUT_DIR).mkdir(exist_ok=True)

    results_df = pd.DataFrame(results_rows).sort_values("RMSE").reset_index(drop=True)
    results_df.to_csv(Path(OUTPUT_DIR) / "metrics.csv", index=False)

    horizon_df = pd.concat(horizon_frames, ignore_index=True)
    horizon_df.to_csv(Path(OUTPUT_DIR) / "metrics_by_horizon.csv", index=False)

    return results_df, horizon_df


def create_eda_plots(raw_df: pd.DataFrame, feature_df: pd.DataFrame):
    Path(OUTPUT_DIR).mkdir(exist_ok=True)

    # Plot 1: top countries by final cumulative value
    latest = raw_df.sort_values("Date").groupby("Country_Region", as_index=False).tail(1)
    top_countries = latest.nlargest(10, "Doses_admin").sort_values("Doses_admin")
    plt.figure(figsize=(10, 6))
    plt.barh(top_countries["Country_Region"], top_countries["Doses_admin"])
    plt.xlabel("Final cumulative doses")
    plt.ylabel("Country")
    plt.title("Top 10 countries by final cumulative vaccine doses")
    plt.tight_layout()
    plt.savefig(Path(OUTPUT_DIR) / "eda_top_countries.png", dpi=200)
    plt.close()

    # Plot 2: example time series for 5 countries
    sample_countries = latest.nlargest(5, "Doses_admin")["Country_Region"].tolist()
    plt.figure(figsize=(11, 6))
    for country in sample_countries:
        temp = raw_df[raw_df["Country_Region"] == country]
        plt.plot(temp["Date"], temp["Doses_admin"], label=country)
    plt.xlabel("Date")
    plt.ylabel("Cumulative doses")
    plt.title("Vaccination trends for selected countries")
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(OUTPUT_DIR) / "eda_time_series.png", dpi=200)
    plt.close()

    # Plot 3: distribution of target_1
    plt.figure(figsize=(10, 6))
    plt.hist(feature_df["target_1"], bins=40)
    plt.xlabel("Target day 1 value")
    plt.ylabel("Frequency")
    plt.title("Distribution of forecast target (day 1)")
    plt.tight_layout()
    plt.savefig(Path(OUTPUT_DIR) / "eda_target_distribution.png", dpi=200)
    plt.close()


def create_comparison_plots(metrics_df: pd.DataFrame, horizon_df: pd.DataFrame):
    Path(OUTPUT_DIR).mkdir(exist_ok=True)

    # Plot 4: overall RMSE by model
    temp = metrics_df.sort_values("RMSE")
    plt.figure(figsize=(9, 5))
    plt.bar(temp["model"], temp["RMSE"])
    plt.ylabel("RMSE")
    plt.xlabel("Model")
    plt.title("Overall RMSE by model")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(Path(OUTPUT_DIR) / "model_rmse_comparison.png", dpi=200)
    plt.close()

    # Plot 5: RMSE by horizon day
    plt.figure(figsize=(10, 6))
    for model_name in horizon_df["model"].unique():
        temp = horizon_df[horizon_df["model"] == model_name]
        plt.plot(temp["horizon_day"], temp["RMSE"], marker="o", label=model_name)
    plt.xlabel("Forecast horizon day")
    plt.ylabel("RMSE")
    plt.title("RMSE by forecast horizon")
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(OUTPUT_DIR) / "rmse_by_horizon.png", dpi=200)
    plt.close()


def main():
    Path(OUTPUT_DIR).mkdir(exist_ok=True)

    print("Loading dataset...")
    raw_df = load_dataset(DATA_PATH)

    print("Creating features...")
    feature_df = create_features(
        raw_df,
        lag_days=LAG_DAYS,
        horizon=HORIZON,
        target_mode=TARGET_MODE
    )

    print("Splitting dataset...")
    train_df, test_df, cutoff_date = split_data(feature_df, test_ratio=TEST_RATIO)

    feature_cols, target_cols, numeric_cols, categorical_cols = get_columns()
    X_train = train_df[feature_cols]
    y_train = train_df[target_cols]

    X_test = test_df[feature_cols]
    y_test = test_df[target_cols]

    save_dataset_summary(raw_df, feature_df, cutoff_date)
    create_eda_plots(raw_df, feature_df)

    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    models = build_models(preprocessor)

    print("Training models...")
    trained_models = {}
    for name, model in tqdm(models.items(), desc="Training Models"):
        model.fit(X_train, y_train)
        trained_models[name] = model

    all_results = []
    horizon_frames = []
    all_predictions = {}

    print("Evaluating baseline...")
    baseline_preds = naive_baseline_predictions(X_test)
    baseline_metrics = evaluate_model(y_test, baseline_preds)
    baseline_metrics["model"] = "NaiveBaseline"
    all_results.append(baseline_metrics)

    baseline_horizon = evaluate_by_horizon(y_test, baseline_preds)
    baseline_horizon["model"] = "NaiveBaseline"
    horizon_frames.append(baseline_horizon)
    all_predictions["NaiveBaseline"] = baseline_preds

    print("Evaluating trained models...")
    for name, model in tqdm(trained_models.items(), desc="Evaluation"):
        preds = model.predict(X_test)
        all_predictions[name] = preds

        metrics = evaluate_model(y_test, preds)
        metrics["model"] = name
        all_results.append(metrics)

        horizon_metrics = evaluate_by_horizon(y_test, preds)
        horizon_metrics["model"] = name
        horizon_frames.append(horizon_metrics)

    metrics_df, horizon_df = save_metrics(all_results, horizon_frames)
    save_predictions(test_df, y_test, all_predictions)
    create_comparison_plots(metrics_df, horizon_df)

    print("\nDone.")
    print(f"Saved outputs to: {OUTPUT_DIR}")
    print("Files created:")
    print("- dataset_summary.csv")
    print("- missing_values_summary.csv")
    print("- metrics.csv")
    print("- metrics_by_horizon.csv")
    print("- predictions.csv")
    print("- eda_top_countries.png")
    print("- eda_time_series.png")
    print("- eda_target_distribution.png")
    print("- model_rmse_comparison.png")
    print("- rmse_by_horizon.png")


if __name__ == "__main__":
    main()
