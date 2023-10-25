from shiny import App, render, ui
import pandas as pd
from pathlib import Path
from plots import dist_plot, scatter_plot

infile = Path(__file__).parent / "penguins.csv"
df = pd.read_csv(infile)

app_ui = ui.page_fluid(
    ui.panel_title("Hello Penguins!"),
    ui.row(
        ui.input_slider(
            "mass",
            "Mass",
            2000,
            8000,
            6000,
        ),
    ),
    ui.row(
        ui.layout_column_wrap(
            1 / 2,
            ui.card(
                ui.input_checkbox("trend", "Add trendline"),
                ui.output_plot("scatter"),
                height="300px",
            ),
            ui.card(
                ui.output_plot("dist"),
                height="300px",
            ),
        )
    ),
)


def server(input):
    @render.plot
    def dist():
        filtered = df.loc[df["body_mass"] < input.mass()]
        return dist_plot(filtered)

    @render.plot
    def scatter():
        filtered = df.loc[df["body_mass"] < input.mass()]
        return scatter_plot(filtered, input.trend())


app = App(app_ui, server)
