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
    gr.inputs.Number(label="Age (years)"),
    gr.inputs.Number(label="Weight (Kg)"),
    gr.inputs.Number(label="Height (Cm)"),
    gr.inputs.Number(label="BMI"),
    gr.inputs.Radio(choices=[0, 1], type="index", label="Fast Food (Y/N)"),
    gr.inputs.Radio(choices=[0, 1], type="index", label="Reg. Exercise (Y/N)"),
    gr.inputs.Radio(choices=[0, 1], type="index", label="Pregnant (Y/N)"),
    gr.inputs.Radio(choices=[0, 1], type="index", label="Skin Darkening (Y/N)"),
    gr.inputs.Radio(choices=[0, 1], type="index", label="Hair Growth (Y/N)"),
    gr.inputs.Radio(choices=[0, 1], type="index", label="Weight Gain (Y/N)"),
    gr.inputs.Radio(choices=[0, 1], type="index", label="Pimples (Y/N)"),
    gr.inputs.Radio(choices=[0, 1], type="index", label="Hair Loss (Y/N)"),
    gr.inputs.Number(label="No. of abortions"),
    gr.inputs.Number(label="Cycle length (days)")
]

outputs = gr.outputs.Textbox(label="PCOS Prediction")

# Create the Gradio interface
demo = gr.Interface(fn=predict_pcos, inputs=inputs, outputs=outputs)

# Launch the app
if __name__ == "__main__":
    demo.launch()
