import librosa
import numpy as np
import pickle

# Dummy model (you can replace later with trained model)
def predict_emotion(file_path):
    try:
        y, sr = librosa.load(file_path, duration=3)

        # Extract feature (simple)
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr).T, axis=0)

        # Dummy logic (for demo)
        value = np.mean(mfcc)

        if value > 0:
            return "Happy 😊"
        else:
            return "Sad 😢"

    except:
        return "Error processing audio"