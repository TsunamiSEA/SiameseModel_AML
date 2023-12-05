import os
import cv2
import shutil

ROOT = 'C:/Users/101234758/Desktop/MLP/original/classification_data/train_data'
CASCADE_PATH = 'C:/Users/101234758/Desktop/MLP/haarcascade_frontalface_default.xml'
CASCADE_MODEL = cv2.CascadeClassifier(CASCADE_PATH)
OUTPUT_DIR = 'Extracted Faces'

# Create the output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

folders = sorted(os.listdir(ROOT))

def extract_face(image):
    """ Gets the face from the image passed using Haar-Cascade """
    global CASCADE_MODEL
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Getting co-ordinates of the face
    faces = CASCADE_MODEL.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=3,
        minSize=(16, 16)
    )
    
    # Extracts the face from the image and returns it
    if len(faces) > 0:
        x, y, w, h = faces[-1]
        image = image[y:y+h, x:x+w]
        return image
    return None

def save_image(path, image):
    """ Saves the Image to the path specified """
    
    # Creating directory
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    # Saving the image
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = cv2.resize(image, (64, 64))
    cv2.imwrite(path, image)

for folder in folders:
    # Gets list of all files in the folder
    files = os.listdir(os.path.join(ROOT, folder))
    
    num_files = 0
    no_face_detected = True  # Assume no face detected initially
    
    for file in files:
        # Reads the image and extracts face
        path = os.path.join(ROOT, folder, file)
        image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        face = extract_face(image)
        
        # Saves the face
        if not face is None:
            save_path = os.path.join(OUTPUT_DIR, folder, f"{num_files}.jpg")
            save_image(save_path, face)
            num_files += 1
            no_face_detected = False  # Face detected, update flag
    
    # If no face is detected in any image within the folder, stop the session
    if no_face_detected:
        print(f"No face detected in folder: {folder}")
        break

# If no face is detected in any folder, the session won't reach this point
shutil.make_archive("Extracted Faces", 'zip', OUTPUT_DIR)
