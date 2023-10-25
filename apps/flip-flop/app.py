from shiny import Inputs, Outputs, Session, App, reactive, render, req, ui
import random

app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.input_radio_buttons(
            "choice", "Which slider?", choices=["Slider 1", "Slider 2"]
        ),
        ui.input_slider("slider_1", "Slider 1", 0, 100, 15),
        ui.input_slider("slider_2", "Slider 2", 0, 100, 45),
    ),
    ui.card({"style": "font-size: larger"}, ui.output_text("txt")),
)


def server(input: Inputs, output: Outputs, session: Session):
    @render.text
    def txt():
        random_number = random.randint(0, 100)
        if input.choice() == "Slider 1":
            return f"Slider 1 is {input.slider_1()}, a random number is {random_number}"
        else:
            return f"Slider 2 is {input.slider_2()}, a random number is {random_number}"


app = App(app_ui, server)
