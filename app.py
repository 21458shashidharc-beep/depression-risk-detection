# app.py

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# ✅ NEW IMPORTS (VOICE FEATURE)
from voice import predict_emotion
import os

app = Flask(__name__)
CORS(app)

# ===============================
# 📊 STEP 1: CREATE DATASET
# ===============================

data = {
    "screen_time": [8,7,6,5,4,3,2,9,8,7,2,3,4,5,6],
    "sleep_hours": [4,5,6,7,8,6,7,3,4,5,8,7,6,5,4],
    "activity_level": [1,2,3,4,5,6,7,1,2,3,6,5,4,3,2],
    "social_interaction": [1,2,3,4,5,6,7,1,2,3,6,5,4,3,2],
    "risk": [2,2,1,1,0,0,0,2,2,1,0,0,1,1,2]
}

df = pd.DataFrame(data)

X = df.drop("risk", axis=1)
y = df["risk"]

# ===============================
# 🤖 STEP 2: TRAIN MODEL
# ===============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier()
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)

print("✅ Model trained successfully!")
print("📊 Accuracy:", accuracy)

# ===============================
# 🔮 STEP 3: PREDICTION API
# ===============================

@app.route('/predict', methods=['POST'])
def predict():

    data = request.get_json()

    screen = float(data['screen_time'])
    sleep = float(data['sleep_hours'])
    activity = float(data['activity_level'])
    social = float(data['social_interaction'])

    input_data = np.array([[screen, sleep, activity, social]])

    prediction = model.predict(input_data)[0]

    if prediction == 2:
        risk = "High Risk 🔴"
        advice = "Seek support, reduce stress, talk to someone ❤️"
    elif prediction == 1:
        risk = "Moderate Risk 🟡"
        advice = "Maintain balance, sleep well, stay active 🙂"
    else:
        risk = "Low Risk 🟢"
        advice = "Keep up healthy habits! 😊"

    return jsonify({
        "risk": risk,
        "advice": advice
    })

# ===============================
# 📊 STEP 4: METRICS API (NEW)
# ===============================

@app.route('/metrics', methods=['GET'])
def metrics():
    return jsonify({
        "accuracy": round(accuracy * 100, 2)
    })

# ===============================
# 🎤 STEP 5: VOICE API (NEW)
# ===============================

@app.route('/voice', methods=['POST'])
def voice():

    file = request.files['audio']

    path = "temp.wav"
    file.save(path)

    result = predict_emotion(path)

    os.remove(path)

    return jsonify({
        "emotion": result
    })

# ===============================
# 🚀 RUN SERVER
# ===============================

if __name__ == '__main__':
    app.run(debug=True)