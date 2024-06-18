import gradio as gr
import joblib
import numpy as np

# Load the models
rf_model = joblib.load("rf_patients.pkl")
scaler = joblib.load("scaler_patients.pkl")

def predict_pcos(age, weight, height, bmi, fast_food, exercise, pregnant, skin_darkening, hair_growth, weight_gain, pimples, hair_loss, abortions, cycle_length):
    # Create a numpy array from the input
    input_data = np.array([[age, weight, height, bmi, fast_food, exercise, pregnant, skin_darkening, hair_growth, weight_gain, pimples, hair_loss, abortions, cycle_length]])
    
    # Scale the input data
    scaled_input = scaler.transform(input_data)
    
    # Make a prediction using the RF model
    prediction = rf_model.predict(scaled_input)
    
    # Convert the prediction to a human-readable label
    if prediction == 0:
        return "No PCOS"
    else:
        return "PCOS Detected"

# Define the input and output interfaces
inputs = [
    gr.Number(label="Age (years)"),
    gr.Number(label="Weight (Kg)"),
    gr.Number(label="Height (Cm)"),
    gr.Number(label="BMI"),
    gr.Radio(choices=[0, 1], label="Fast Food (Y/N)"),
    gr.Radio(choices=[0, 1], label="Reg. Exercise (Y/N)"),
    gr.Radio(choices=[0, 1], label="Pregnant (Y/N)"),
    gr.Radio(choices=[0, 1], label="Skin Darkening (Y/N)"),
    gr.Radio(choices=[0, 1], label="Hair Growth (Y/N)"),
    gr.Radio(choices=[0, 1], label="Weight Gain (Y/N)"),
    gr.Radio(choices=[0, 1], label="Pimples (Y/N)"),
    gr.Radio(choices=[0, 1], label="Hair Loss (Y/N)"),
    gr.Number(label="No. of abortions"),
    gr.Number(label="Cycle length (days)")
]

outputs = gr.Textbox(label="PCOS Prediction")

# Create the Gradio interface
demo = gr.Interface(fn=predict_pcos, inputs=inputs, outputs=outputs)

# Launch the app
if __name__ == "__main__":
    demo.launch()
