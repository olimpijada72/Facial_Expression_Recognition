import cv2
import dlib

# Path to the image file
image_path = 'data/me_pic.jpeg'  # Change this to the path of your image file

# Load the image
image = cv2.imread(image_path)
if image is None:
    print("Could not read the image.")
    exit()

# Convert image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Initialize dlib's face detector (HOG-based) and facial landmarks predictor
hog_face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Detect faces in the grayscale image
faces = hog_face_detector(gray)

# Loop over each detected face
for face in faces:
    # Get the bounding box of the face
    x1 = face.left()
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()
    
    # Draw a rectangle around the face
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Get the landmarks of the face
    face_landmarks = dlib_facelandmark(gray, face)
    
    print("")
    print("#############")
    print("LANDMARKS")
    print("#############")
    print("")
    print("     x      y")

    # Loop over each landmark point
    for n in range(0, 68):
        x = face_landmarks.part(n).x
        y = face_landmarks.part(n).y
        print(n+1,': ', x, y)

        
        # Draw a circle for each landmark point
        cv2.circle(image, (x, y), 1, (0, 255, 255), 1)

# Display the output image with face bounding boxes and landmarks
cv2.imshow("Face Landmarks", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
 
print()