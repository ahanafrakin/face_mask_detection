# using open CV 2 for face detection
import cv2

'''
    The method of face detection used will be haar cascade classifiers
    This is an ML approach that uses a cascade function that is trained from positive and 
    negative images to detect objects in images. 
    
    The xml file saved in the this working directory contains all the data needed to train open cv Cascade 
    Classifier
    
'''

# Process and load the haar cascade
face_cascades = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Get a live video from the webcam, face detection improves with better lighting so ensure the room is well lit
# possible disadvantage is capturing video at night since reduced lighting reduces accuracy
live_video = cv2.VideoCapture(0)
