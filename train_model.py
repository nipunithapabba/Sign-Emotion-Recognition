import os
import numpy as np
import pickle 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. SETUP PATHS AND ACTIONS
DATA_PATH = os.path.join('MP_Data') 
actions = np.array(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'])
label_map = {label:num for num, label in enumerate(actions)}

sequences, labels = [], []

print("Loading 3D Hand Landmark Data...")

for action in actions:
    action_folder = os.path.join(DATA_PATH, action)
    if not os.path.exists(action_folder):
        continue

    num_samples = len(os.listdir(action_folder))
    print(f"Adding {num_samples} samples for: {action}")
    
    for sequence in range(num_samples):
        try:
            # Loading your 3D [.npy] files (x, y, z)
            res = np.load(os.path.join(action_folder, str(sequence), "0.npy"))
            sequences.append(res)
            labels.append(label_map[action])
        except:
            continue

# 2. DATA PREPARATION
X = np.array(sequences)
y = np.array(labels)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# 3. THE "SUPREME COURT" MODEL (2000 Trees + 3D Optimization)
model = RandomForestClassifier(
    n_estimators=1600, 
    criterion='entropy',      # Entropy is better for many classes (26 letters)
    max_features='sqrt',      # Forces trees to find the most important features (like Z-depth)
    class_weight='balanced',   # Handles your 30 vs 60 sample difference perfectly
    oob_score=True,            # Cross-validation during training
    n_jobs=-1,                 # Speed: Use all CPU cores
    random_state=42
)

# 4. TRAIN
print("\nTraining the 26-letter 3D Brain...")
model.fit(X_train, y_train)

# 5. FINAL ESTIMATION
y_pred = model.predict(X_test)
test_acc = accuracy_score(y_test, y_pred) * 100
oob_acc = model.oob_score_ * 100

print("-" * 40)
print(f"FINAL TEST ACCURACY: {test_acc:.2f}%")
print(f"OOB ESTIMATION (How it handles new data): {oob_acc:.2f}%")
print("-" * 40)

# 6. SAVE EVERYTHING
# We save 'actions' inside the pickle so test_model.py is always in sync
with open('sign_model.p', 'wb') as f:
    pickle.dump({'model': model, 'actions': actions}, f)

print("Project complete! Your Sign Language Brain is ready for deployment.")