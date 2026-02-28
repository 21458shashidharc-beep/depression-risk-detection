import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import pickle

# =========================
# 📊 LOAD DATASET
# =========================
data = pd.read_csv('../dataset/data.csv')

X = data.drop('risk', axis=1)
y = data['risk']

# =========================
# 🔀 TRAIN TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# =========================
# 🤖 TRAIN MODEL
# =========================
model = RandomForestClassifier(n_estimators=200)
model.fit(X_train, y_train)

# =========================
# 📈 PREDICTION
# =========================
y_pred = model.predict(X_test)

# =========================
# 📊 METRICS
# =========================
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

print("✅ Model trained successfully!")
print("📊 Accuracy:", accuracy)
print("📊 Precision:", precision)
print("📊 Recall:", recall)
print("📊 F1 Score:", f1)

# =========================
# 💾 SAVE MODEL
# =========================
pickle.dump(model, open('../model/model.pkl', 'wb'))

# =========================
# 🌈 COLORFUL BAR GRAPH
# =========================
values = [accuracy, precision, recall, f1]
labels = ["Accuracy", "Precision", "Recall", "F1-Score"]

colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#f9ca24']

plt.figure(figsize=(7,5))

bars = plt.bar(labels, values, color=colors, width=0.5)

plt.ylim(0, 1)

plt.title("📊 Model Performance Metrics", fontsize=14)
plt.ylabel("Score")

# Show values on top
for bar in bars:
    y = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2,
             y + 0.02,
             round(y, 2),
             ha='center',
             fontsize=10)

plt.grid(axis='y', linestyle='--', alpha=0.5)

# ✅ SAVE IMAGE
plt.savefig("performance_graph.png")

plt.show()

# =========================
# 📊 CONFUSION MATRIX
# =========================
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5,5))

plt.imshow(cm, cmap='Blues')
plt.title("Confusion Matrix")

plt.xlabel("Predicted")
plt.ylabel("Actual")

for i in range(len(cm)):
    for j in range(len(cm[i])):
        plt.text(j, i, cm[i][j], ha='center')

plt.colorbar()

# ✅ SAVE IMAGE
plt.savefig("confusion_matrix.png")

plt.show()