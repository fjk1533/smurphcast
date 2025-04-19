from __future__ import annotations
import typer, json, sys
import pandas as pd
from pathlib import Path

from .pipeline import ForecastPipeline
from .explain.explainer import ForecastExplainer
from .evaluation.backtest import rolling_backtest

app = typer.Typer(add_completion=False, help="SmurphCast CLI")


def _load_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        typer.echo(f"❌ File not found: {path}", err=True)
        raise typer.Exit(1)


@app.command()
def fit(
    data: Path = typer.Argument(..., exists=True, readable=True),
    horizon: int = typer.Option(..., help="Forecast horizon"),
    model: str = typer.Option("additive", help="additive | gbm | beta_rnn"),
    save: Path = typer.Option("model.pkl", help="Path to save fitted pipeline"),
):
    """Fit a model and persist it to a Pickle."""
    df = _load_csv(data)
    pipe = ForecastPipeline(model_name=model).fit(df, horizon=horizon)
    pipe.save(save)
    typer.echo(f"✅ Saved model to {save}")


@app.command()
def predict(
    model_path: Path = typer.Argument(..., exists=True),
    output: Path = typer.Option("forecast.csv"),
    interval: bool = typer.Option(False, help="Include prediction interval."),
):
    """Load a saved model and produce its next-horizon forecast."""
    pipe = ForecastPipeline.load(model_path)
    if interval:
        df_ci = pipe.predict_interval()
        df_ci.to_csv(output, index=True)
    else:
        forecast = pipe.predict()
        forecast.to_csv(output, header=True)
    typer.echo(f"✅ Forecast saved to {output}")



@app.command()
def backtest(
    data: Path = typer.Argument(..., exists=True),
    horizon: int = typer.Option(...),
    splits: int = typer.Option(3),
    model: str = typer.Option("additive"),
):
    """Rolling-origin back‑test; prints MAE per fold + overall."""
    df = _load_csv(data)

    def _fit(d): return ForecastPipeline(model_name=model).fit(d, horizon=horizon)
    res = rolling_backtest(
        fit_fn=_fit,
        predict_fn=lambda m, h: m.predict(),
        df=df,
        horizon=horizon,
        splits=splits,
    )
    maes = res["mae_per_fold"]
    typer.echo(json.dumps({"mae_per_fold": maes, "mean_mae": sum(maes) / len(maes)}, indent=2))


@app.command()
def explain(
    model_path: Path = typer.Argument(..., exists=True),
    top: int = typer.Option(10, help="Top‑k features to display"),
):
    """Plot feature importances for a saved model."""
    pipe = ForecastPipeline.load(model_path)
    exp = ForecastExplainer(pipe._model)
    fi = exp.feature_importance().head(top)
    typer.echo(fi.to_string())
    try:
        exp.contribution_plot(k=top)
    except Exception:
        typer.echo("📊 Plot skipped (no GUI backend).")


if __name__ == "__main__":
    app()
