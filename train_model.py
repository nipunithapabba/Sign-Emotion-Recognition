import os
import numpy as np
import pickle # Used to save the model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. LOAD DATA
DATA_PATH = os.path.join('MP_Data') 
# Top of train_model.py
actions = np.array(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'])
label_map = {label:num for num, label in enumerate(actions)}

sequences, labels = [], []
for action in actions:
    for sequence in range(30):
        # Load the data we collected earlier
        res = np.load(os.path.join(DATA_PATH, action, str(sequence), "0.npy"))
        sequences.append(res)
        labels.append(label_map[action])

# 2. PREPROCESS
X = np.array(sequences)
y = np.array(labels)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# 3. BUILD THE MODEL (Random Forest)
# This is a different type of AI that is great for landmark data
# Change from 100 to 1000 trees
model = RandomForestClassifier(n_estimators=1000, max_depth=None, min_samples_split=2)

# 4. TRAIN
print("AI is learning (Random Forest style)...")
model.fit(X_train, y_train)

# 5. TEST
y_pred = model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred) * 100}%")

# 6. SAVE
with open('sign_model.p', 'wb') as f:
    pickle.dump({'model': model}, f)

print("Success! 'sign_model.p' has been created.")