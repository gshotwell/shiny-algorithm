from shiny import Inputs, Outputs, Session, App, ui, render, reactive, req
import pandas as pd
from pathlib import Path
from plots import (
    plot_score_distribution,
    plot_auc_curve,
    plot_precision_recall_curve,
    plot_api_response,
)

file_path = Path(__file__).parent / "simulated-data.csv"


@reactive.file_reader(file_path, interval_secs=0.2)
def df():
    out = pd.read_csv(file_path)
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    return out


training_tab = ui.nav(
    "Training Dashboard",
    ui.row(
        ui.layout_column_wrap(
            1 / 2,
            ui.card(
                ui.card_header("Model Metrics"),
                ui.output_plot("metric"),
                ui.input_select(
                    "metric",
                    "Metric",
                    choices=["ROC Curve", "Precision-Recall"],
                ),
            ),
            ui.card(
                ui.card_header("Training Scores"),
                ui.output_plot("score_dist"),
            ),
        ),
    ),
)

app_ui = ui.page_navbar(
    training_tab,
    sidebar=ui.sidebar(
        ui.input_select(
            "account",
            "Account",
            choices=[
                "Berge & Berge",
                "Fritsch & Fritsch",
                "Hintz & Hintz",
                "Mosciski and Sons",
                "Wolff Ltd",
            ],
        ),
        width="300px",
    ),
    id="tabs",
    title="Training",
)


def server(input: Inputs, output: Outputs, session: Session):
    # Training dashboard

    @render.plot
    def score_dist():
        df_value = df()
        df_filtered = df_value[df_value["account"] == input.account()]
        return plot_score_distribution(df_filtered)

    @render.plot
    def metric():
        df_value = df()
        df_filtered = df_value[df_value["account"] == input.account()]
        if input.metric() == "ROC Curve":
            return plot_auc_curve(df_filtered, "is_electronics", "training_score")
        else:
            return plot_precision_recall_curve(
                df_filtered, "is_electronics", "training_score"
            )


app = App(app_ui, server)
