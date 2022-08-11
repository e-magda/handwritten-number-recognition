import numpy as np
import streamlit as st
import cv2
import pickle
import math
from PIL import Image
from scipy import ndimage


def drawing_cleaner(img):
    """Function making the drawn image comparable to training images"""
    img = Image.open("image.jpg")
    size = 28, 28
    img.thumbnail(size,Image.ANTIALIAS)
    img.save("image_28px.jpg")
    img = img.convert('L')  # convert image to 8-bit grayscale

    img = list(img.getdata()) # convert image data to a list of integers
    # convert that to 2D list (list of lists of integers)
    img = [img[offset:offset+28] for offset in range(0, 28*28, 28)]

    img = np.array(img)
    img = np.where(img > 50, img, 0)

    # Condition to check if canvas is not empty (if only 0s corresponding to black color)
    if np.count_nonzero(img) == 0:
        return ""

    # remove every row and column at the sides of the image which are completely black
    while np.sum(img[0]) == 0:
        img = img[1:]

    while np.sum(img[:,0]) == 0:
        img = np.delete(img,0,1)

    while np.sum(img[-1]) == 0:
        img = img[:-1]

    while np.sum(img[:,-1]) == 0:
        img = np.delete(img,-1,1)

    rows,cols = img.shape

    img = img.astype('float32')

    if rows > cols:
        factor = 20.0/rows
        rows = 20
        cols = int(round(cols*factor))
        img = cv2.resize(img, (cols,rows))
    else:
        factor = 20.0/cols
        cols = 20
        rows = int(round(rows*factor))
        img = cv2.resize(img, (cols,rows))

    #Adding the missing black rows and columns at each side to get back to a 28x28px image
    colsPadding = (int(math.ceil((28-cols)/2.0)),int(math.floor((28-cols)/2.0)))
    rowsPadding = (int(math.ceil((28-rows)/2.0)),int(math.floor((28-rows)/2.0)))
    img = np.lib.pad(img,(rowsPadding,colsPadding),'constant')

    # Using functions to shift the image as required
    shiftx, shifty = getBestShift(img)
    img = shift(img, shiftx, shifty)

    # Conversion to required format (1, 748) for our model
    img = img.flatten()
    img = np.array(img).reshape(1, -1) 
    img = img.astype('int64')

    return img


def classifier(img):
    """Function predicting the numbers"""
    if img != "":
        img = drawing_cleaner(img)

        # Condition for empty canvas (checking if only 0s corresponding to black color)
        if np.count_nonzero(img) == 0:
            return ""

        # Loading the model and saving the result
        loaded_model = pickle.load(open('digit_classifier.sav', 'rb'))    
        result_1 = loaded_model.predict(img)[0]

        # Calculating class probabilities 
        probabilities = loaded_model.predict_proba(img)
        probabilities = probabilities[0]
        indices = (-probabilities).argsort()

        probability_1 = probabilities[indices[0]]
        probability_1 = np.round(probability_1 * 100, 2)

        # Saving also 2nd most likely class for our number in case the first prediction is incorrect
        result_2 = indices[1]
        probability_2 = probabilities[indices[1]]
        probability_2 = np.round(probability_2 * 100, 2)

    return [result_1, probability_1, result_2, probability_2]


def display_result(img):
    """Function displaying the results of the classifier"""

    # In case the canvas is empty, we return nothing in the app
    if len(classifier(img)) < 4:
        return ""

    else:
        result_1, probability_1, result_2, probability_2 = classifier(img)
        st.subheader(f"You've drawn {result_1}. Correct?")
        st.text(f"Certainty: {probability_1} %")
        st.text(f"Otherwise it could be {result_2} with a certainty of {probability_2} %")


def getBestShift(img):
    """Function shifting the inner box of the image so that it is centered using the center of mass"""
    cy,cx = ndimage.measurements.center_of_mass(img)

    rows,cols = img.shape
    shiftx = np.round(cols/2.0-cx).astype(int)
    shifty = np.round(rows/2.0-cy).astype(int)

    return shiftx,shifty


def shift(img,sx,sy):
    """Function shifting the image in the given directions"""
    rows,cols = img.shape
    M = np.float32([[1,0,sx],[0,1,sy]])
    shifted = cv2.warpAffine(img,M,(cols,rows))
    return shifted