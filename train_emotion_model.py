import os
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

# ============================================================
# SETTINGS
# ============================================================
DATA_PATH = 'Emotion_Data'
MODEL_SAVE_PATH = 'emotion_model.p'

# ============================================================
# LOAD DATA — picks up all emotions dynamically
# (Happy/Sad/Surprised = 100 samples, Angry/Neutral = 150)
# ============================================================
emotions = sorted([e for e in os.listdir(DATA_PATH)
                   if os.path.isdir(os.path.join(DATA_PATH, e))])
label_map = {emotion: idx for idx, emotion in enumerate(emotions)}

print(f"Found {len(emotions)} emotions: {emotions}")
print("Loading face landmark data...\n")

sequences, labels = [], []

for emotion in emotions:
    folder = os.path.join(DATA_PATH, emotion)
    samples = [f for f in os.listdir(folder) if f.endswith('.npy')]
    print(f"  {emotion}: {len(samples)} samples")
    for s in samples:
        try:
            data = np.load(os.path.join(folder, s))
            sequences.append(data)
            labels.append(label_map[emotion])
        except:
            continue

X = np.array(sequences)
y = np.array(labels)

print(f"\nTotal samples: {len(X)}")
print(f"Feature size: {X.shape[1]}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

# ============================================================
# MODEL
# ============================================================
model = MLPClassifier(
    hidden_layer_sizes=(512, 256, 128),
    activation='relu',
    solver='adam',
    max_iter=2000,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
    verbose=True
)

print(f"\nTraining on {len(X_train)} samples...")
model.fit(X_train, y_train)

# ============================================================
# EVALUATE
# ============================================================
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred) * 100

print("\n" + "=" * 40)
print(f"FINAL TEST ACCURACY: {accuracy:.2f}%")
print("=" * 40)
print("\nPer-emotion breakdown:")
print(classification_report(y_test, y_pred, target_names=emotions))

with open(MODEL_SAVE_PATH, 'wb') as f:
    pickle.dump({'model': model, 'emotions': emotions}, f)

print(f"\n✅ Saved as '{MODEL_SAVE_PATH}'")