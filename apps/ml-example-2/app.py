from shiny import Inputs, Outputs, Session, App, ui, render, reactive, req
import pandas as pd
from pathlib import Path
from plots import (
    plot_score_distribution,
    plot_auc_curve,
    plot_precision_recall_curve,
    plot_api_response,
)


df = pd.read_csv(Path(__file__).parent / "simulated-data.csv")
df["date"] = pd.to_datetime(df["date"], errors="coerce")

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

annotation_tab = ui.nav(
    "Data Annotation",
    ui.row(
        ui.layout_column_wrap(
            1 / 2,
            ui.card(ui.card_header("Results"), ui.output_data_frame("results")),
            ui.card(
                ui.card_header("Annotate"),
                ui.output_text("to_review"),
                ui.card_footer(
                    ui.layout_column_wrap(
                        1 / 2,
                        ui.input_action_button(
                            "is_electronics", "Electronics", class_="btn btn-primary"
                        ),
                        ui.input_action_button(
                            "not_electronics",
                            "Not Electronics",
                            class_="btn btn-secondary",
                        ),
                    ),
                ),
            ),
        )
    ),
)

app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.input_select(
            "account",
            "Account",
            choices=df["account"].unique().tolist(),
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
    ),
    ui.navset_bar(
        training_tab,
        monitoring_tab,
        annotation_tab,
        title="Options",
        id="tabs",
    ),
)


def server(input: Inputs, output: Outputs, session: Session):
    # Training dashboard

    @render.plot
    def score_dist():
        df_filtered = df[df["account"] == input.account()]
        return plot_score_distribution(df_filtered)

    @render.plot
    def metric():
        df_filtered = df[df["account"] == input.account()]
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
        out = df[(df["date"] > start_date) & (df["date"] <= end_date)].sample(
            n=input.sample(), replace=True
        )
        return out

    @reactive.Calc()
    def filtered_data():
        sample_df = sampled_data()
        sample_df = sample_df.loc[sample_df["account"] == input.account()]
        return sample_df

    @render.plot
    def api_response():
        return plot_api_response(filtered_data())

    @render.plot
    def prod_score_dist():
        return plot_score_distribution(filtered_data())

    # Annotation tab

    @render.data_frame
    def results():
        return render.DataGrid(
            filtered_data()[["text", "prod_score"]],
            width="100%",
            row_selection_mode="single",
            filters=True,
        )

    @render.text
    def to_review():
        selected_row = list(req(input.results_selected_rows()))
        print(selected_row)
        print(filtered_data()["text"][selected_row[0]])
        if selected_row:
            return filtered_data()["text"].iloc[selected_row[0]]
        else:
            return ""


app = App(app_ui, server)
