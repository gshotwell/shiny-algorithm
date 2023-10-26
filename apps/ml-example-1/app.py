from shiny import Inputs, Outputs, Session, App, ui, render
import pandas as pd
from pathlib import Path
from plots import dist_plot, plot_auc_curve, plot_precision_recall_curve


df = pd.read_csv(Path(__file__).parent / "simulated-data.csv")

app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.input_select("account", "Account", choices=df["account"].unique().tolist())
    ),
    ui.navset_bar(
        ui.nav(
            "Training dashboard",
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
                        ui.card_header("Score distribution"),
                        ui.output_plot("score_dist"),
                    ),
                ),
            ),
        ),
        title="Options",
    ),
)


def server(input: Inputs, output: Outputs, session: Session):
    @render.plot
    def score_dist():
        df_filtered = df[df["account"] == input.account()]
        return dist_plot(df_filtered)

    @render.plot
    def metric():
        df_filtered = df[df["account"] == input.account()]
        if input.metric() == "ROC Curve":
            return plot_auc_curve(df_filtered, "is_electronics", "training_score")
        else:
            return plot_precision_recall_curve(
                df_filtered, "is_electronics", "training_score"
            )


app = App(app_ui, server)
