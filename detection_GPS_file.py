import argparse
import time
import numpy as np
from PIL import Image
from pycoral.adapters import classify
from pycoral.utils.edgetpu import make_interpreter

def main():
    print("\n")
    print("########################################################")
    print("Welcome to the GPS Signal Interference Detection Program")
    print("########################################################")
    print("\n")
    input("Press Enter to continue...")
    print("\n")
    
    model_path = input("Enter the path of the model .tflite file: ")
    interpreter = make_interpreter(model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    input_shape = input_details[0]['shape']
    num_input_values = input_shape[1]
    print("\n")
    print("The model requires " + str(num_input_values) + " input attributes.")

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
        print("\n*************\nClassification Results for Package:\n*************\n")
        print("Attributes: " + str( input_package[0][:len(input_values)]))
        for c in classes:
            print('Interference: {}, Score: {:.5f}'.format((c.score > 0.5), c.score))
            input("\nPress Enter to continue...")
    input("Press Enter to EXIT...")
        
if __name__ == '__main__':
    main()

