import cv2
import dlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay


# image_path = 'data/fer_2.jpg'
image_path = 'data/random_faces/me_pic2.png'
# image_path = 'data/zaidi.jpeg'


image = cv2.imread(image_path)
if image is None:
    print("Could not read the image.")
    exit()

# Convert image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Initialize dlib's face detector (HOG-based) and facial landmarks predictor
hog_face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

# Detect faces in the grayscale image
faces = hog_face_detector(gray)


for face in faces:
    # Get bounding box 
    x1 = face.left()
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()
    
    # Draw bounding box 
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Get and draw landmarks of the face
    face_landmarks = dlib_facelandmark(gray, face)

    landmarks = np.empty((0, 2), int)
    for n in range(0, 68):
        x = face_landmarks.part(n).x
        y = face_landmarks.part(n).y

        landmarks = np.append(landmarks, [[x, y]], axis = 0)  #storing in numpy array for Delaunay triang
        cv2.circle(image, (x, y), 1, (0, 255, 255), 1)
    
    print(landmarks)

    #Computing and drawing Delaunay triangulation 
    tri = Delaunay(landmarks)
    counter= 0
    for triangle in tri.simplices:
        #Takes the coordinates of vertices and reshapes them to work for polylines input
        # print(tri.simplices)
        pts = landmarks[triangle].reshape(-1, 1, 2)

        cv2.polylines(image, [pts], isClosed=True, color=(255, 0, 0), thickness= 1)

    



# Display the output image with face bounding boxes and landmarks
cv2.imshow("Face Landmarks", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
