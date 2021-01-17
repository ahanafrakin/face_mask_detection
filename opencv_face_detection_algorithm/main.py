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

# Continuously loop reading frames until I press an escape key
while True:
    # Get the frames from live video, return code tells us if we're done scanning frames but with webcam
    # we won't run out
    return_code, frame = live_video.read()

    # OpenCV uses grayscale so we convert the frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect the faces in the cascade, returns a list of rectangles where faces are supposedly found
    faces = face_cascades.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    # Draw a rectangle around a face, (x,y) top left, w : width, h : height
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Video', frame)

    # quit if keyboard interrupt...wait for 'q' to be pressed
    k = cv2.waitKey(1) & 0xff == ord('q')
    if k == 27:
        break

# Release the captures when quit
live_video.release()
cv2.destroyAllWindows()
