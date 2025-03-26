import gradio as gr
import pandas as pd
import pickle

# Load trained model
with open("apartment_model.pkl", "rb") as f:
    model = pickle.load(f)

# Define prediction function
def predict_density(population, tax_income):
    # Calculate the engineered feature
    tax_per_person = tax_income / population

    # Prepare input data
    input_df = pd.DataFrame([{
        "pop": population,
        "tax_income": tax_income,
        "tax_per_person": tax_per_person
    }])

    # Predict using the trained model
    prediction = model.predict(input_df)[0]
    return (
        f"Estimated population density: {prediction:.2f} people/km²\n"
        f"Tax per person: {tax_per_person:.2f} CHF"
    )

# Build Gradio interface
demo = gr.Interface(
    fn=predict_density,
    inputs=[
        gr.Number(label="Population"),
        gr.Number(label="Total Tax Income (CHF)")
    ],
    outputs="text",
    title="Population Density Predictor",
    description="Predicts municipality population density based on total tax income and population size. Uses engineered feature: tax per person.",
    examples=[
        [10000, 12500000],
        [2000, 1500000],
        [50000, 62000000]
    ]
)

# Launch the app
demo.launch()
