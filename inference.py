import argparse
import os
import cv2
import numpy as np
import tensorflow as tf

label_dictionary = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6',
 7: '7', 8: '8', 9: '9', 10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F',
 16: 'G', 17: 'H', 18: 'I', 19: 'J', 20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O',
 25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T', 30: 'U', 31: 'V', 32: 'W', 33: 'X',
 34: 'Y', 35: 'Z', 36: 'a', 37: 'b', 38: 'd', 39: 'e', 40: 'f', 41: 'g', 42: 'h',
 43: 'n', 44: 'q', 45: 'r', 46: 't'}


def main(input_dir):
    # Load the pre-trained model
    model = tf.keras.models.load_model('model')

    # Get a list of image files in the input directory
    image_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Loop over the image files and predict their labels
    for image_file in image_files:
        # Load the image and preprocess it for the model
        img = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (28, 28))
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=-1)
        img = np.expand_dims(img, axis=0)

        # Predict the label of the image
        probs = model.predict(img, verbose=0)
        label = np.argmax(probs)
        ascii = str(ord(label_dictionary[label])).zfill(3)
        #letter = label_dictionary[label] # if you need not ascii but letter

        # Print the predicted label and file path to the console
        print(f'{ascii}, {os.path.relpath(image_file, input_dir)}')


if __name__ == '__main__':
    # Parse the command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Path to directory containing input images')
    args = parser.parse_args()

    # Call the main function with the input directory
    main(args.input)
