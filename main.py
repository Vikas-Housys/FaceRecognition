# main.py
import os
import sys
import customtkinter as ctk
from face_utils import FaceProcessor
from ui_components import FaceRecognitionUI
from model_downloader import download_models

def main():
    # Create necessary directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("images", exist_ok=True)
    
    # Download required models if they don't exist
    model_files = [
        "shape_predictor_68_face_landmarks.dat",
        "dlib_face_recognition_resnet_model_v1.dat"
    ]
    
    for model in model_files:
        if not os.path.exists(f"models/{model}"):
            print(f"Downloading {model}...")
            download_models(model)
    
    # Initialize the application
    app = ctk.CTk()
    app.title("Face Recognition System")
    app.geometry("800x600")
    
    # Initialize face processor
    face_processor = FaceProcessor()
    
    # Create and run UI
    ui = FaceRecognitionUI(app, face_processor)
    app.mainloop()

if __name__ == "__main__":
    main()

