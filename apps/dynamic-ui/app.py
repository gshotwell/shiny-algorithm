from shiny import Inputs, Outputs, Session, App, reactive, render, req, ui

choices = [
    "ui.input_text('txt_input', 'A text input')",
    "ui.input_numeric('number', 'A numeric input', value = 0.5)",
]
app_ui = ui.page_fluid(
    ui.input_select("selector", "Select an input", choices=choices),
    ui.output_ui("result"),
    ui.output_text("txt"),
)


def server(input: Inputs, output: Outputs, session: Session):
    @output
    @render.ui
    def result():
        input_str = input.selector()
        input_obj = eval(input_str)
        return input_obj

    @output
    @render.text
    def txt():
        if input.selector() == choices[0]:
            return f"You entered {input.txt_input()}"
        else:
            return f"The number is {input.number()}"


app = App(app_ui, server)
