# Neural network for recognition of handwritten letters and numbers

# Description

This project is a simple neural network that can recognize handwritten characters (letters and numbers) using the EMNIST dataset. It also includes a script that can be used to classify handwritten character images and output their corresponding ASCII values.

`images` - test data

`model` - our trained model

`inference.py` - python file that takes one CLI argument that is path to
directory with image samples and print output to console. Script all images in directory in common formats like PNG or JPG

`requirements.txt` - required libraries for venv

`train.ipynb` - jupyter notebook file that was created on the kaggle which contains the code to create the model

# Dependencies

You can install all dependencies using pip. For example:

> > pip install -r requirements.txt

# Usage

## Training the Model

To train neural network you should have an account on Kaggle. The model trains on this site

## Classifying Handwritten Character Images

To classify handwritten character images, you can run the `inference.py` script. This script takes one command line argument, which is the path to the directory containing the images. The output will be printed to the console in the format [character ASCII index in decimal format], [path to image sample].

Example:

> > python inference.py --input images

output:

```
067, letter_C.jpg
068, letter_D.png
070, letter_F.png
072, letter_H.png
078, letter_N.jpg
051, number_3.png
053, number_5.png
```
"# CHI_ML_Internship_2023" 
