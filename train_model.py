import pandas as pd
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# ================= PATH SETUP =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

data_path = os.path.join(BASE_DIR, 'data', 'dataset.csv')
training_path = os.path.join(BASE_DIR, 'data', 'dataset.csv')


def load_clean_dataset(csv_path):
    dataset = pd.read_csv(csv_path)
    dataset = dataset.loc[:, ~dataset.columns.str.contains(r'^Unnamed:')]
    dataset = dataset.loc[:, dataset.columns.notna()]
    dataset = dataset.loc[:, dataset.columns != '']
    dataset = dataset.dropna(axis=1, how='all')
    dataset = dataset.loc[:, ~dataset.columns.duplicated()]

    target_column = 'prognosis' if 'prognosis' in dataset.columns else 'disease'
    return dataset, target_column

# Create models folder
models_dir = os.path.join(BASE_DIR, 'models')
os.makedirs(models_dir, exist_ok=True)

# ================= RANDOM FOREST (MAIN MODEL) =================
dataset, target_column = load_clean_dataset(training_path)
X = dataset.drop(target_column, axis=1)
y = dataset[target_column]

le = LabelEncoder()
Y_encoded = le.fit_transform(y)

rf_model = RandomForestClassifier()
rf_model.fit(X, Y_encoded)

# Save RF model
rf_path = os.path.join(models_dir, 'model.pkl')
pickle.dump(rf_model, open(rf_path, 'wb'))

print("✅ RandomForest model saved at models/model.pkl")

# ================= SVC MODEL =================

# Split
x_train, x_test, y_train, y_test = train_test_split(
    X, Y_encoded, test_size=0.3, random_state=20
)

svc_model = SVC(kernel='linear', probability=True)
svc_model.fit(x_train, y_train)

# Save SVC model
svc_path = os.path.join(models_dir, 'svc.pkl')
pickle.dump(svc_model, open(svc_path, 'wb'))

encoder_path = os.path.join(models_dir, 'label_encoder.pkl')
pickle.dump(le, open(encoder_path, 'wb'))

print("✅ SVC model saved at models/svc.pkl")


# # Load datasets
# df = pd.read_csv('../data/dataset.csv')
# severity = pd.read_csv('../data/Symptom-severity.csv')

# df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
# df = df.loc[:, ~df.columns.duplicated()]
# # Create severity dictionary
# severity_dict = dict(zip(severity['Symptom'], severity['weight']))

# # Replace 1s with severity values
# for col in df.columns:
#     if col != 'disease':
#         df[col] = df[col].apply(lambda x: severity_dict.get(col, 1) if x == 1 else 0)

# # Split data
# X = df.drop('disease', axis=1)
# y = df['disease']

# # Train model
# model = RandomForestClassifier()
# model.fit(X, y)

# # Save model
# pickle.dump(model, open('model.pkl', 'wb'))

# print("Model trained with severity!")

# print("Accuracy of SVC model:", accuracy_score(y_test, svc_model.predict(x_test)))
# print("Accuracy of RF model:", accuracy_score(Y_encoded, rf_model.predict(X)))
# print("Accuracy of RF model on test set:", accuracy_score(y_test, rf_model.predict(x_test)))
# print("Accuracy of SVC model on training set:", accuracy_score(Y_encoded, svc_model.predict(X)))
