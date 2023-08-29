import argparse
import time
import numpy as np
from PIL import Image
from pycoral.adapters import classify
from pycoral.utils.edgetpu import make_interpreter

def get_input_values(input_shape):
    values = []
    for i in range(input_shape[1]):
        value = float(input(f"Enter value {i+1}: "))
        values.append(value)
    return values

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
    print("\n")
    print(f"The model requires {input_shape[1]} input attributes.")
    input_values = get_input_values(input_shape)
    input("Press Enter to continue...")    
    top_k = 5
    threshold = 0.0
    
    input_values = np.array(input_values, dtype=np.float32).reshape(input_shape)
    interpreter.set_tensor(input_details[0]['index'], input_values)

    # Run inference
    interpreter.invoke()
    classes = classify.get_classes_from_scores(classify.get_scores(interpreter), top_k, threshold)

    # Display the results
    print("\n*************\nClassification Results:")
    for c in classes:
        print('Interference: {}, Score: {:.5f}'.format((c.score > 0.5), c.score))
    input("\nPress enter to EXIT...")
    
if __name__ == '__main__':
    main()
