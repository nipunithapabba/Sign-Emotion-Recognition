import os
import numpy as np
import pickle 
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier # The new "Brain"
from sklearn.metrics import accuracy_score

# 1. SETUP PATHS AND ACTIONS
DATA_PATH = os.path.join('MP_Data') 

# Dynamically find which letters you have recorded (A, B, C, D, E...)
recorded_actions = [action for action in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, action))]
actions = np.array(sorted(recorded_actions))
label_map = {label:num for num, label in enumerate(actions)}

sequences, labels = [], []

print(f"Detected {len(actions)} letters: {actions}")
print("Loading Normalized 3D Hand Landmark Data...")

for action in actions:
    action_folder = os.path.join(DATA_PATH, action)
    samples = [s for s in os.listdir(action_folder) if os.path.isdir(os.path.join(action_folder, s))]
    print(f"Adding {len(samples)} samples for: {action}")
    
    for sample in samples:
        try:
            res = np.load(os.path.join(action_folder, sample, "0.npy"))
            sequences.append(res)
            labels.append(label_map[action])
        except:
            continue

# 2. DATA PREPARATION
X = np.array(sequences)
y = np.array(labels)

if len(X) == 0:
    print("Error: No data found! Check your MP_Data folder.")
    exit()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# 3. THE MULTI-LAYER PERCEPTRON (Neural Network)
# This model is much better at finding patterns in "Relative" (normalized) coordinates
model = MLPClassifier(
    hidden_layer_sizes=(128, 64, 32), # 3 Layers of neurons
    max_iter=2000,                   # Give it plenty of time to learn
    activation='relu',               # Standard "on/off" switch for neurons
    solver='adam',                   # Fast learning algorithm
    random_state=42,
    verbose=True                     # This lets you watch the "Loss" go down as it learns
)

# 4. TRAIN
print(f"\nNeural Network is training on {len(X)} samples...")
model.fit(X_train, y_train)

# 5. FINAL ESTIMATION
y_pred = model.predict(X_test)
test_acc = accuracy_score(y_test, y_pred) * 100

print("-" * 40)
print(f"SUCCESS: Trained on {len(actions)} letters")
print(f"FINAL TEST ACCURACY: {test_acc:.2f}%")
print("-" * 40)

# 6. SAVE EVERYTHING
with open('sign_model.p', 'wb') as f:
    pickle.dump({'model': model, 'actions': actions}, f)

print("Neural Network 'sign_model.p' is ready for deployment!")