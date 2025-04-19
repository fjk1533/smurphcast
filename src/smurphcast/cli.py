import typer
from .pipeline import ForecastPipeline
import pandas as pd

app = typer.Typer(help="SmurphCast – forecast bounded KPI time series")

@app.command("fit")
def fit_cli(
    csv_path: str,
    horizon: int = typer.Option(..., help="Forecast horizon (number of periods)"),
    model: str = typer.Option("additive", help="additive | gbm | beta_rnn"),
):
    df = pd.read_csv(csv_path)
    pipe = ForecastPipeline(model_name=model).fit(df, horizon)
    forecast = pipe.predict()
    forecast.to_csv("forecast.csv", index=False)
    typer.echo("➜ Forecast saved to forecast.csv")

if __name__ == "__main__":
    app()
