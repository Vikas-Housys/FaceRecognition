# model_downloader.py
import requests
import os

MODEL_URLS = {
    "shape_predictor_68_face_landmarks.dat": "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2",
    "dlib_face_recognition_resnet_model_v1.dat": "http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2"
}

def download_models(model_name):
    """Download the required model files if they don't exist."""
    if model_name not in MODEL_URLS:
        raise ValueError(f"Unknown model: {model_name}")
    
    url = MODEL_URLS[model_name]
    save_path = f"models/{model_name}"
    
    # Download the compressed file
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    # Save and decompress
    import bz2
    with open(save_path, 'wb') as f:
        decompressor = bz2.BZ2Decompressor()
        for data in response.iter_content(chunk_size=8192):
            if data:
                f.write(decompressor.decompress(data))
    
    