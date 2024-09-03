import cv2
import dlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay


image_path = 'data/random_faces/mourinho.jpg'
# image_path = 'data/me_pic.jpeg'
# image_path = 'data/zaidi.jpeg'


image = cv2.imread(image_path)
if image is None:
    print("Could not read the image.")
    exit()

# Convert image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Initialize face detector and facial landmarks predictor
hog_face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

# Detect faces in the grayscale image
faces = hog_face_detector(gray)

# Define new image on which we will draw mesh
image_size = 224
mesh_image = np.zeros((image_size, image_size), dtype=np.uint8)



def scale_landmarks_to_image(landmarks, image_size):
        # Convert landmarks to a numpy array if it's not already
        landmarks = np.array(landmarks)
        
        # Step 1: Calculate the center of the landmarks
        mean_x = np.mean(landmarks[:, 0])
        mean_y = np.mean(landmarks[:, 1])
        
        # Step 2: Center the landmarks
        centered_landmarks = landmarks - [mean_x, mean_y]
        
        # Step 3: Scale to fit within image_size
        max_abs_coords = np.max(np.abs(centered_landmarks))
        scale_factor = image_size*0.97 / (2 * max_abs_coords)
        scaled_landmarks = centered_landmarks * scale_factor
        
        # Adjust to be in the range [0, image_size]
        scaled_landmarks += image_size / 2
        
        return scaled_landmarks

for face in faces:

    # Get landmarks of the face
    face_landmarks = dlib_facelandmark(gray, face)

    landmarks = np.empty((0, 2), int)
    for n in range(0, 68):
        x = face_landmarks.part(n).x
        y = face_landmarks.part(n).y

         #storing in numpy array for Delaunay triang
        landmarks = np.append(landmarks, [[x, y]], axis = 0) 

    landmarks = scale_landmarks_to_image(landmarks, image_size)

    #Computing and drawing Delaunay triangulation 
    tri = Delaunay(landmarks)



    for triangle in tri.simplices:
        pts = landmarks[triangle].reshape(-1, 1, 2).astype(np.int32)
        cv2.polylines(mesh_image, [pts], isClosed=True, color=(255, 0, 0), thickness= 1)



# Display the output mesh image and save it
cv2.imshow("Face Landmarks", mesh_image)
cv2.imwrite("test/mourinho_mesh.jpg", mesh_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
