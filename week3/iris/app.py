import gradio as gr
import pandas as pd
from sklearn.datasets import load_iris
import pickle
 
# Load the improved model
model_filename = "improved_iris_model.pkl"
model = pickle.load(open(model_filename, 'rb'))
 
# Load iris dataset for target names
iris = load_iris()
 
# Extended prediction function to handle new feature
def predict(sepal_length, sepal_width, petal_length, petal_width):
    # Calculate the new feature
    sepal_ratio = sepal_length / sepal_width
    # Prepare the input DataFrame including the new feature
    input_df = pd.DataFrame([[
        sepal_length,
        sepal_width,
        petal_length,
        petal_width,
        sepal_ratio
    ]], columns=[
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)",
        "sepal_ratio"
    ])
 
    # Load the improved model
    model_filename = "improved_iris_model.pkl"
    model = pickle.load(open(model_filename, 'rb'))
    # Make prediction
    prediction = model.predict(input_df)[0]
    iris = load_iris()
    return iris.target_names[prediction]
 
# Gradio interface
demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Number(label="Sepal Length (cm)"),
        gr.Number(label="Sepal Width (cm)"),
        gr.Number(label="Petal Length"),
        gr.Number(label="Petal Width")
    ],
    outputs="text",
    examples=[
        [5.1, 3.5, 1.4, 0.2],
        [6.2, 2.9, 4.3, 1.3],
        [7.7, 3.8, 6.7, 2.2],
    ],
    title="Enhanced Iris Flower Prediction",
    description="Enter sepal and petal measurements to predict the Iris species (enhanced model with additional feature: Sepal Ratio)."
)
 
demo.launch()