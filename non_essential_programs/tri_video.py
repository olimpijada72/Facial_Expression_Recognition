import dlib
import cv2

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay


cap = cv2.VideoCapture(0)

#loading models
hog_face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

while True:
    _, frame = cap.read()

    #converting to grayscale bcs of easier computation
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #find faces
    faces = hog_face_detector(gray)

    for face in faces:

        # Getting and drawing bounding box
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        face_landmarks = dlib_facelandmark(gray, face)


        # Getting and drawing landmarks
        landmarks = np.empty((0, 2), int)
        for n in range(0,68):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
           
            #storing in numpy array for Delaunay triang
            landmarks = np.append(landmarks, [[x, y]], axis = 0) 
            cv2.circle(frame, (x, y), 1, (0, 255, 255), 1)

        #Computing and drawing Delaunay triangulation 
        tri = Delaunay(landmarks)
        for triangle in tri.simplices:
            pts = landmarks[triangle].reshape(-1, 1, 2)
            cv2.polylines(frame, [pts], isClosed=True, color=(255, 0, 0), thickness=1)



    cv2.imshow("Face Landmarks", frame)


    key = cv2.waitKey(1)
    #Pres ESC to close window
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()