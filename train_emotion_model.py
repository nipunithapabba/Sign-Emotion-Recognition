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
# LOAD DATA
# ============================================================
emotions = sorted([e for e in os.listdir(DATA_PATH)
                   if os.path.isdir(os.path.join(DATA_PATH, e))])
label_map = {emotion: idx for idx, emotion in enumerate(emotions)}

print(f"Found {len(emotions)} emotions: {emotions}")
print("Loading face landmark data...\n")

sequences, labels = [], []

for emotion in emotions:
    emotion_folder = os.path.join(DATA_PATH, emotion)
    samples = [f for f in os.listdir(emotion_folder) if f.endswith('.npy')]
    print(f"  Loading {len(samples)} samples for: {emotion}")
    for sample_file in samples:
        try:
            data = np.load(os.path.join(emotion_folder, sample_file))
            sequences.append(data)
            labels.append(label_map[emotion])
        except:
            continue

X = np.array(sequences)
y = np.array(labels)

if len(X) == 0:
    print("Error: No data found!")
    exit()

print(f"\nTotal samples: {len(X)}")
print(f"Feature size: {X.shape[1]}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

# ============================================================
# MODEL
# Bigger network + more iterations for higher accuracy
# ============================================================
model = MLPClassifier(
    hidden_layer_sizes=(512, 256, 128),  # Deeper network than before
    activation='relu',
    solver='adam',
    max_iter=2000,                        # More room to converge
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,                  # More patient before stopping
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

# ============================================================
# SAVE
# ============================================================
with open(MODEL_SAVE_PATH, 'wb') as f:
    pickle.dump({'model': model, 'emotions': emotions}, f)

print(f"\n✅ Emotion model saved as '{MODEL_SAVE_PATH}'")