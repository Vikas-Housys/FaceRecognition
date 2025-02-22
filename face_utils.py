# face_utils.py
import cv2
import dlib
import numpy as np
import pandas as pd
import os
from datetime import datetime

class FaceProcessor:
    def __init__(self, min_face_size=160, max_face_size=320):
        # Initialize dlib's face detector and facial landmarks predictor
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
        self.face_rec = dlib.face_recognition_model_v1("models/dlib_face_recognition_resnet_model_v1.dat")
        self.data_file = "face_data.csv"
        self.known_faces = self.load_known_faces()
        
        # Face size range parameters
        self.min_face_size = min_face_size  # minimum face width in pixels
        self.max_face_size = max_face_size  # maximum face width in pixels

    def load_known_faces(self):
        """Load known faces from CSV file."""
        if os.path.exists(self.data_file):
            df = pd.read_csv(self.data_file)
            return {name: np.array(eval(encoding)) for name, encoding in 
                   zip(df['label'], df['face_encoding'])}
        return {}

    def get_face_encoding(self, image):
        """Extract face encoding from image."""
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        faces = self.detector(rgb_image)
        if not faces:
            return None, "No face detected"
        
        # Check face size
        face = faces[0]
        face_width = face.right() - face.left()
        
        if face_width < self.min_face_size:
            return None, "Face too far away"
        if face_width > self.max_face_size:
            return None, "Face too close"
        
        # Get facial landmarks and compute face encoding
        shape = self.predictor(rgb_image, face)
        face_encoding = np.array(self.face_rec.compute_face_descriptor(rgb_image, shape))
        
        return face_encoding, "Success"

    def register_face(self, name, image):
        """Register a new face."""
        face_encoding, message = self.get_face_encoding(image)
        if face_encoding is None:
            return False, message
        
        # Save image
        os.makedirs(f"images/{name}", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cv2.imwrite(f"images/{name}/{timestamp}.jpg", image)
        
        # Save encoding
        if name not in self.known_faces:
            self.known_faces[name] = face_encoding
            self.save_to_csv(name, face_encoding)
        
        return True, "Success"

    def save_to_csv(self, name, encoding):
        """Save face encoding to CSV file."""
        new_data = pd.DataFrame({
            'label': [name],
            'face_encoding': [encoding.tolist()],
            'timestamp': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
        })
        
        if os.path.exists(self.data_file):
            df = pd.read_csv(self.data_file)
            df = pd.concat([df, new_data], ignore_index=True)
        else:
            df = new_data
            
        df.to_csv(self.data_file, index=False)

    def recognize_face(self, image, threshold=0.6):
        """Recognize face in image."""
        face_encoding, message = self.get_face_encoding(image)
        if face_encoding is None:
            return None, message
        
        # Compare with known faces
        for name, known_encoding in self.known_faces.items():
            distance = np.linalg.norm(face_encoding - known_encoding)
            if distance < threshold:
                return name, "Success"
                
        return "Unknown", "Face not recognized"
    
    
    