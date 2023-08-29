import argparse
import time
import numpy as np
from PIL import Image
from pycoral.adapters import classify
from pycoral.utils.edgetpu import make_interpreter
from flask import Flask, render_template

app = Flask(__name__)

output_text = ""

def log_output(text):
    global output_text
    output_text += text + "\n"

def main():
    log_output("\n")
    log_output("########################################################")
    log_output("Welcome to the GPS Signal Interference Detection Program")
    log_output("########################################################")
    log_output("\n")
    
    model_path = "modelo_1.tflite"
    interpreter = make_interpreter(model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    input_shape = input_details[0]['shape']
    num_input_values = input_shape[1]
    log_output("\n")
    log_output("The model requires " + str(num_input_values) + " input attributes.")

    # Read input values from the file
    input_values = []
    with open("signals.txt", "r") as file:
        for line in file:
            value = float(line.strip())
            input_values.append(value)
    
    top_k = 5
    threshold = 0.0
    
    # Iterate through input_values in packages of num_input_values
    for i in range(0, len(input_values), num_input_values):
        input_package = input_values[i:i+num_input_values]

        # Pad or truncate the input_values to match the required input_shape
        input_package = input_package + [0.0] * (num_input_values - len(input_package))
        
        input_package = np.array([input_package], dtype=np.float32)  # Wrap in a batch
        interpreter.set_tensor(input_details[0]['index'], input_package)

        # Run inference
        interpreter.invoke()
        classes = classify.get_classes_from_scores(classify.get_scores(interpreter), top_k, threshold)

        # Display the results
        log_output("\n*************\nClassification Results for Package:\n*************\n")
        log_output("Attributes: " + str( input_package[0][:len(input_values)]))
        for c in classes:
            log_output('Interference: {}, Score: {:.5f}'.format((c.score > 0.5), c.score))

        
@app.route("/")
def index():
    return render_template("index.html", output=output_text)

if __name__ == '__main__':
    main()
    app.run(debug=True)
