import os
import numpy as np
import pickle 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. LOAD DATA
DATA_PATH = os.path.join('MP_Data') 
actions = np.array(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'J', 'Z'])
label_map = {label:num for num, label in enumerate(actions)}

sequences, labels = [], []

print("Starting data load...")

for action in actions:
    action_folder = os.path.join(DATA_PATH, action)
    
    # This automatically counts if you have 30 or 60 samples
    num_samples = len(os.listdir(action_folder))
    print(f"Loading {num_samples} samples for letter: {action}")
    
    for sequence in range(num_samples):
        try:
            res = np.load(os.path.join(action_folder, str(sequence), "0.npy"))
            sequences.append(res)
            labels.append(label_map[action])
        except Exception as e:
            print(f"Could not load sequence {sequence} for {action}: {e}")

# 2. PREPROCESS
X = np.array(sequences)
y = np.array(labels)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# 3. BUILD THE MODEL
# 1000 trees to handle the increased complexity of 17 letterss
model = RandomForestClassifier(
    n_estimators=1600, 
    max_depth=None, 
    min_samples_leaf=1, 
    max_features='log2', # This forces the trees to look at fewer features at a time, finding hidden patterns
    random_state=42
)

# 4. TRAIN
print("\nAI is learning (Random Forest style)...")
model.fit(X_train, y_train)

# 5. TEST
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred) * 100
print(f"Model Accuracy: {accuracy:.2f}%")

# 6. SAVE
with open('sign_model.p', 'wb') as f:
    pickle.dump({'model': model, 'actions': actions}, f) # Saving actions too for safety

print("Success! 'sign_model.p' has been created for 17 letters.")