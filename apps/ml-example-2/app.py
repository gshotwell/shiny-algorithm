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

monitoring_tab = ui.nav(
    "Model Monitoring",
    ui.row(
        ui.layout_column_wrap(
            1 / 2,
            ui.card(
                ui.card_header("API Response Time"),
                ui.output_plot("api_response"),
            ),
            ui.card(
                ui.card_header("Production Scores"),
                ui.output_plot("prod_score_dist"),
            ),
        ),
    ),
)

app_ui = ui.page_navbar(
    training_tab,
    monitoring_tab,
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
        ui.panel_conditional(
            "input.tabs !== 'Training Dashboard'",
            ui.input_date_range(
                "dates",
                "Dates",
                start="2023-01-01",
                end="2023-04-01",
            ),
            ui.input_numeric("sample", "Sample Size", value=10000, step=5000),
        ),
        width="300px",
    ),
    id="tabs",
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

    # Model monitoring

    @reactive.Calc
    def sampled_data() -> pd.DataFrame:
        start_date, end_date = input.dates()
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        df_value = df()
        out = df_value[
            (df_value["date"] > start_date) & (df_value["date"] <= end_date)
        ].sample(n=input.sample(), replace=True)
        return out

    @reactive.Calc()
    def filtered_data() -> pd.DataFrame:
        sample_df = sampled_data()
        sample_df = sample_df.loc[sample_df["account"] == input.account()]
        return sample_df.reset_index(drop=True)

    @render.plot
    def api_response():
        return plot_api_response(filtered_data())

    @render.plot
    def prod_score_dist():
        return plot_score_distribution(filtered_data())


app = App(app_ui, server)
