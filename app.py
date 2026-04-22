from flask import Flask, request, render_template
from flask_sqlalchemy import SQLAlchemy
import pickle
import numpy as np
import pandas as pd
import ast
import re
import json
import tensorflow as tf
from werkzeug.utils import secure_filename
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.applications.mobilenet_v2 import decode_predictions
from tensorflow.keras.preprocessing import image
import os
import csv
from flask import session, redirect, url_for
from werkzeug.security import generate_password_hash, check_password_hash
from flask import session, redirect


sym_des = pd.read_csv("data/symtoms_df.csv")
precautions_df = pd.read_csv("data/precautions_df.csv")
workout_df = pd.read_csv("data/workout_df.csv")
description_df = pd.read_csv("data/description.csv")
medications_df = pd.read_csv("data/medications.csv")
diets_df = pd.read_csv("data/diets.csv")

os.makedirs('static/uploads', exist_ok=True)
svc = pickle.load(open('models/svc.pkl', 'rb'))


def load_clean_dataset(csv_path):
    dataset = pd.read_csv(csv_path)
    dataset = dataset.loc[:, ~dataset.columns.str.contains(r'^Unnamed:')]
    dataset = dataset.loc[:, dataset.columns.notna()]
    dataset = dataset.loc[:, dataset.columns != '']
    dataset = dataset.dropna(axis=1, how='all')
    dataset = dataset.loc[:, ~dataset.columns.duplicated()]

    target_column = 'prognosis' if 'prognosis' in dataset.columns else 'disease'
    return dataset, target_column


training_df, target_column = load_clean_dataset('data/dataset.csv')
columns = list(getattr(svc, 'feature_names_in_', training_df.drop(columns=[target_column]).columns))


app = Flask(__name__)   # ✅ must come BEFORE routes
app.secret_key = "secret123"

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# ================== DATABASE ==================
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    
    name = db.Column(db.String(100))
    mobile = db.Column(db.String(20), unique=True)
    email = db.Column(db.String(100))
    password = db.Column(db.String(100))

    gender = db.Column(db.String(20))
    age = db.Column(db.String(10))
    height = db.Column(db.String(10))
    weight = db.Column(db.String(10))
    blood_group = db.Column(db.String(10))

    conditions = db.Column(db.String(300))
    smoking = db.Column(db.String(10))
    alcohol = db.Column(db.String(10))
    activity = db.Column(db.String(20))

    medications = db.Column(db.String(300))
    family_history = db.Column(db.String(300))

# create DB
with app.app_context():
    db.create_all()


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'models', 'model.pkl')
model = pickle.load(open(model_path, 'rb'))
label_encoder_path = os.path.join(BASE_DIR, 'models', 'label_encoder.pkl')
label_encoder = pickle.load(open(label_encoder_path, 'rb')) if os.path.exists(label_encoder_path) else None
medical_image_model_path = os.path.join(BASE_DIR, 'models', 'medical_image_model.keras')
medical_image_labels_path = os.path.join(BASE_DIR, 'models', 'medical_image_labels.json')
medical_image_metadata_path = os.path.join(BASE_DIR, 'models', 'medical_image_metadata.json')

medical_image_model = None
medical_image_labels = []
medical_image_metadata = {}

if os.path.exists(medical_image_model_path) and os.path.exists(medical_image_labels_path):
    medical_image_model = tf.keras.models.load_model(medical_image_model_path)
    with open(medical_image_labels_path, 'r', encoding='utf-8') as f:
        medical_image_labels = json.load(f)
    if os.path.exists(medical_image_metadata_path):
        with open(medical_image_metadata_path, 'r', encoding='utf-8') as f:
            medical_image_metadata = json.load(f)
else:
    image_model = MobileNetV2(weights='imagenet')

severity = pd.read_csv('data/Symptom-severity.csv')
severity_dict = dict(zip(severity['Symptom'], severity['weight']))

# ================== DICTIONARIES ==================
symptoms_dict = {symptom: index for index, symptom in enumerate(columns)}

DISEASE_ALIASES = {
    'diabetes': 'Diabetes',
    'dengue': 'Dengue',
    'pneumonia': 'Pneumonia',
    'hepatitas': 'hepatitis A',
    'hepatitis': 'hepatitis A',
}


def normalize_name(value):
    return " ".join(str(value).strip().lower().split())


def normalize_text(value):
    cleaned = re.sub(r'[^a-z0-9\s]', ' ', str(value).lower().replace('_', ' '))
    return " ".join(cleaned.split())


def parse_list_field(value):
    if isinstance(value, list):
        return value
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    except (ValueError, SyntaxError):
        pass
    return [text]


def resolve_disease_name(disease, values):
    normalized_values = {normalize_name(value): value for value in values if str(value).strip()}
    normalized_disease = normalize_name(disease)

    if normalized_disease in normalized_values:
        return normalized_values[normalized_disease]

    alias = DISEASE_ALIASES.get(normalized_disease)
    if alias:
        normalized_alias = normalize_name(alias)
        if normalized_alias in normalized_values:
            return normalized_values[normalized_alias]

    for normalized_value, original_value in normalized_values.items():
        if normalized_disease in normalized_value or normalized_value in normalized_disease:
            return original_value

    return disease


def helper(disease):
    description_name = resolve_disease_name(disease, description_df['Disease'].dropna().tolist())
    precautions_name = resolve_disease_name(disease, precautions_df['Disease'].dropna().tolist())
    medications_name = resolve_disease_name(disease, medications_df['Disease'].dropna().tolist())
    diets_name = resolve_disease_name(disease, diets_df['Disease'].dropna().tolist())
    workout_name = resolve_disease_name(disease, workout_df['disease'].dropna().tolist())

    # Description
    desc = description_df[description_df['Disease'] == description_name]['Description']
    desc = " ".join(desc.values)

    # Precautions
    pre = precautions_df[precautions_df['Disease'] == precautions_name][
        ['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']
    ]
    pre = pre.values.tolist()

    # Medications
    med = medications_df[medications_df['Disease'] == medications_name]['Medication'].values.tolist()

    # Diet
    diet = diets_df[diets_df['Disease'] == diets_name]['Diet'].values.tolist()

    # Workout
    workout = workout_df[workout_df['disease'] == workout_name]['workout'].values.tolist()

    return desc, pre, med, diet, workout


def build_disease_symptom_map():
    symptom_frame = training_df.copy()
    symptom_frame[target_column] = symptom_frame[target_column].astype(str)
    disease_symptoms = {}

    for disease_name, group in symptom_frame.groupby(target_column):
        feature_means = group[columns].mean(numeric_only=True).sort_values(ascending=False)
        top_features = [feature for feature, score in feature_means.items() if score > 0][:6]
        disease_symptoms[normalize_name(disease_name)] = top_features

    return disease_symptoms


DISEASE_SYMPTOMS = build_disease_symptom_map()


def get_disease_info(disease):
    desc, precautions, meds, diet, workout = helper(disease)
    return {
        'description': desc,
        'precautions': [item for item in (precautions[0] if precautions else []) if str(item).strip()],
        'medications': [entry for item in meds for entry in parse_list_field(item) if str(entry).strip()],
        'diet': [entry for item in diet for entry in parse_list_field(item) if str(entry).strip()],
        'workout': [item for item in workout if str(item).strip()],
        'symptoms': DISEASE_SYMPTOMS.get(normalize_name(disease), []),
    }


KNOWN_DISEASE_NAMES = sorted({
    *[str(value) for value in description_df['Disease'].dropna().tolist()],
    *[str(value) for value in precautions_df['Disease'].dropna().tolist()],
    *[str(value) for value in medications_df['Disease'].dropna().tolist()],
    *[str(value) for value in diets_df['Disease'].dropna().tolist()],
    *[str(value) for value in workout_df['disease'].dropna().tolist()],
    *[str(value) for value in training_df[target_column].dropna().tolist()],
}, key=len, reverse=True)


def find_disease_in_message(message):
    normalized_message = normalize_text(message)

    for alias, actual in DISEASE_ALIASES.items():
        if normalize_text(alias) in normalized_message:
            return actual

    for disease in KNOWN_DISEASE_NAMES:
        if normalize_text(disease) in normalized_message:
            return disease

    return None


def extract_symptoms_from_message(message):
    normalized_message = normalize_text(message)
    matched_symptoms = []

    for symptom in columns:
        options = {
            normalize_text(symptom),
            normalize_text(symptom.replace('_', ' ')),
        }
        if any(option and option in normalized_message for option in options):
            matched_symptoms.append(symptom)

    return matched_symptoms


def detect_chat_intents(message):
    normalized_message = normalize_text(message)

    intent_map = {
        'description': ['what is', 'description', 'about', 'explain', 'tell me about', 'overview'],
        'precautions': ['precaution', 'prevent', 'careful', 'avoid', 'safety'],
        'medications': ['medicine', 'medication', 'drug', 'tablet', 'treatment'],
        'diet': ['diet', 'food', 'eat', 'meal', 'nutrition'],
        'workout': ['workout', 'exercise', 'activity', 'fitness'],
        'symptoms': ['symptom', 'sign', 'feel', 'having', 'have', 'suffering'],
    }

    matched = {intent for intent, keywords in intent_map.items() if any(keyword in normalized_message for keyword in keywords)}
    return matched or {'summary'}


def format_items(items):
    return ", ".join(items) if items else "not available in the current dataset"


def build_chatbot_reply(message):
    normalized_message = normalize_text(message)

    if any(greeting in normalized_message for greeting in ['hello', 'hi', 'hey', 'good morning', 'good evening']):
        return (
            "Hello! I can help with disease descriptions, symptoms, precautions, medicines, diet, and workout tips "
            "from your medical datasets. You can ask things like 'what are the precautions for dengue?' or "
            "'I have fever and fatigue'."
        )

    disease = find_disease_in_message(message)
    matched_symptoms = extract_symptoms_from_message(message)
    explicit_disease = disease is not None

    if not disease and matched_symptoms:
        disease = get_predicted_value(matched_symptoms)

    if not disease:
        return (
            "I couldn't match that to a disease in the current dataset yet. Try asking with a disease name like "
            "'diabetes' or symptoms like 'fever, fatigue, vomiting', and I can share the description, precautions, "
            "medicines, diet, and workout guidance."
        )

    info = get_disease_info(disease)
    intents = detect_chat_intents(message)
    asked_for_specific_detail = intents != {'summary'}

    if matched_symptoms and not explicit_disease and not asked_for_specific_detail:
        intents = {'summary'}
    elif matched_symptoms and not explicit_disease and intents == {'symptoms'}:
        intents = {'summary'}

    response_parts = [f"I found information for {disease}. Here's a friendly summary:"]

    if 'summary' in intents or 'description' in intents:
        description = info['description'] or f"{disease} is included in the current dataset, but a description was not found."
        response_parts.append(f"Description: {description}")

    if 'summary' in intents or 'symptoms' in intents:
        symptom_items = [symptom.replace('_', ' ') for symptom in info['symptoms']]
        if matched_symptoms:
            symptom_items = [symptom.replace('_', ' ') for symptom in matched_symptoms]
        response_parts.append(f"Common symptom clues: {format_items(symptom_items)}.")

    if 'summary' in intents or 'precautions' in intents:
        response_parts.append(f"Precautions: {format_items(info['precautions'])}.")

    if 'summary' in intents or 'medications' in intents:
        response_parts.append(f"Medicines often listed in the dataset: {format_items(info['medications'])}.")

    if 'summary' in intents or 'diet' in intents:
        response_parts.append(f"Diet support: {format_items(info['diet'])}.")

    if 'summary' in intents or 'workout' in intents:
        response_parts.append(f"Workout or lifestyle guidance: {format_items(info['workout'])}.")

    response_parts.append("This is supportive information only, so please check with a doctor for diagnosis or treatment.")
    return "\n\n".join(response_parts)



@app.route('/')
def home():
    return render_template('home.html', symptoms=columns)

# ================== ROUTES ==================

# 🔐 AUTH PAGE
@app.route('/auth')
def auth():
    return render_template('auth.html')


@app.route('/signup', methods=['POST'])
def signup():
    name = request.form.get('name')
    mobile = request.form.get('mobile')
    email = request.form.get('email')
    password = request.form.get('password')

    age = request.form.get('age')
    gender = request.form.get('gender')
    height = request.form.get('height')
    weight = request.form.get('weight')
    blood_group = request.form.get('blood_group')

    conditions = request.form.get('conditions')
    smoking = request.form.get('smoking')
    alcohol = request.form.get('alcohol')
    activity = request.form.get('activity')

    medications = request.form.get('medications')
    family_history = request.form.get('family_history')

    # 🔒 Check existing user
    if User.query.filter_by(mobile=mobile).first():
        return render_template('auth.html', error="User already exists")

    # 🔐 Hash password
    hashed_password = generate_password_hash(password)

    new_user = User(
        name=name,
        mobile=mobile,
        email=email,
        password=hashed_password,
        age=age,
        gender=gender,
        height=height,
        weight=weight,
        blood_group=blood_group,
        conditions=conditions,
        smoking=smoking,
        alcohol=alcohol,
        activity=activity,
        medications=medications,
        family_history=family_history
    )

    db.session.add(new_user)
    db.session.commit()

    # ✅ Auto login after signup
    session['user'] = name

    return redirect('/predict-page')   # ✅ go to index/home


# 🔐 LOGIN

@app.route('/login', methods=['POST'])
def login():
    mobile = request.form.get('mobile')
    password = request.form.get('password')

    user = User.query.filter_by(mobile=mobile).first()

    if user and check_password_hash(user.password, password):
        session['user'] = user.name
        return redirect('/predict-page')  # ✅ go to index/home
    else:
        return render_template('auth.html', error="Invalid mobile or password")

# 🔓 LOGOUT
@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect('/auth')

@app.route('/predict-page')
def predict_page():
    return render_template('index.html', symptoms=columns)

def get_predicted_value(patient_symptoms):
    input_vector = np.zeros(len(symptoms_dict))

    for item in patient_symptoms:
        if item in symptoms_dict:
            input_vector[symptoms_dict[item]] = 1

    input_df = pd.DataFrame([input_vector], columns=columns)
    prediction = svc.predict(input_df)[0]
    if label_encoder is not None:
        return label_encoder.inverse_transform([prediction])[0]
    return prediction

@app.route('/predict', methods=['POST'])
def predict():
    if 'user' not in session:
        return redirect('/auth')

    selected_symptoms = request.form.getlist('symptoms')
    
    if not selected_symptoms:
        return render_template("index.html", message="Please select symptoms")


    user_symptoms = [s.strip().lower() for s in selected_symptoms]

    predicted_disease = get_predicted_value(user_symptoms)

    if predicted_disease == "Invalid":
        return render_template("index.html", message="Invalid symptom entered")

    # Get additional info
    desc, precautions, meds, diet, workout = helper(predicted_disease)

    warning = None

    if 'chest_pain' in selected_symptoms and 'breathlessness' in selected_symptoms:
        warning = "⚠️ High-risk condition. Seek medical attention immediately."

    file = request.files.get('image')

    image_result = None
    image_conf = None
    uploaded_image = None

    if file and file.filename != "":
        filename = secure_filename(file.filename)
        filepath = os.path.join('static/uploads', filename)
        file.save(filepath)
        uploaded_image = filepath

        image_result, image_conf = predict_image(filepath)

    # --- Build explanation ---
    important_symptoms = []

    for symptom in selected_symptoms:
        weight = severity_dict.get(symptom, 1)
        important_symptoms.append((symptom, weight))

    # Sort
    important_symptoms = sorted(important_symptoms, key=lambda x: x[1], reverse=True)

    # Top 3
    important_symptoms = important_symptoms[:3]

    # Convert to readable
    def get_level(weight):
        if weight >= 5:
            return "High"
        elif weight >= 3:
            return "Medium"
        else:
            return "Low"

    important_symptoms = [(s.replace('_', ' '), get_level(w)) for s, w in important_symptoms ]

    explanation_text = f"The system predicted {predicted_disease} because your selected symptoms match the pattern for this condition, especially "

    top_symptoms = [s for s, _ in important_symptoms]

    if len(top_symptoms) >= 2:
        explanation_text += ", ".join(top_symptoms[:2])
    elif len(top_symptoms) == 1:
        explanation_text += top_symptoms[0]
    else:
        explanation_text += "the available symptom profile"

    explanation_text += "."

    # Existing symptom prediction
    input_data = []
    for col in columns:
        if col in selected_symptoms:
            input_data.append(1)
        else:
            input_data.append(0)

    input_df = pd.DataFrame([input_data], columns=columns)

    prediction = model.predict(input_df)[0]
    prob = max(model.predict_proba(input_df)[0])

    final_prediction = (
        label_encoder.inverse_transform([prediction])[0]
        if label_encoder is not None else prediction
    )
    final_confidence = prob

    # 🔥 Fusion Logic
    if image_result:
        if image_result and image_conf:
            if image_conf > 0.7:
                final_confidence = min(prob + 0.15, 1.0)
            elif image_conf > 0.4:
                final_confidence = min(prob + 0.05, 1.0)

    if not selected_symptoms:   
        return render_template(
            'index.html',
            symptoms=columns,
            error="Please select at least one symptom"
        )

    record_file = 'records.csv'

    # Convert symptoms list to string
    symptoms_str = ", ".join(selected_symptoms)

    # Check if file exists
    file_exists = os.path.isfile(record_file)

    with open(record_file, 'a', newline='') as f:
        writer = csv.writer(f)

        # Write header if file is new
        if not file_exists:
            writer.writerow(['Symptoms', 'Prediction', 'Confidence', 'Image'])

        writer.writerow([
            symptoms_str,
            final_prediction,
            round(final_confidence * 100, 2),
            image_result if image_result else "None"
        ])
    return render_template(
        'index.html',
        symptoms=columns,
        result=final_prediction,
        confidence=round(final_confidence * 100, 2),
        image_result=image_result,
        image_conf=round(image_conf * 100, 2) if image_conf else None,
        uploaded_image=uploaded_image,
        warning=warning,
        important_symptoms=important_symptoms,
        explanation=explanation_text,
        predicted_disease=predicted_disease,
        dis_des=desc,
        my_precautions=precautions[0] if precautions else [],
        medications=meds,
        my_diet=diet,
        workout=workout 
    )


def predict_image(img_path):
    target_size = tuple(medical_image_metadata.get('image_size', [224, 224]))
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    if medical_image_model is not None and medical_image_labels:
        img_array = efficientnet_preprocess(img_array.copy())
        preds = medical_image_model.predict(img_array, verbose=0)[0]
        top_index = int(np.argmax(preds))
        label = medical_image_labels[top_index]
        confidence = float(preds[top_index])
        return label.replace('_', ' '), confidence

    img_array = img_array / 255.0

    preds = image_model.predict(img_array)
    decoded = decode_predictions(preds, top=1)[0][0]

    label = decoded[1]
    confidence = decoded[2]

    # Optional simple filtering (safe version)
    medical_keywords = ['skin', 'rash', 'lesion', 'wound']

    if any(word in label.lower() for word in medical_keywords):
        return label, confidence
    else:
        return "Not a medical image", confidence

@app.route('/records')
def view_records():
    try:
        df = pd.read_csv('records.csv')
        data = df.to_dict(orient='records')
    except:
        data = []

    return render_template('records.html', records=data)
    
@app.route('/chat', methods=['GET', 'POST'])
def chat():
    chat_history = session.get('chat_history', [])

    if request.method == 'POST':
        user_msg = request.form.get('message', '').strip()
        if user_msg:
            response = build_chatbot_reply(user_msg)
            chat_history.append({'role': 'user', 'text': user_msg})
            chat_history.append({'role': 'assistant', 'text': response})
            session['chat_history'] = chat_history[-12:]

    return render_template('chat.html', chat_history=session.get('chat_history', []))
    
@app.route('/terms')    
def terms():
    return render_template('terms.html')
@app.route('/profile')
def profile():
    return render_template('profile.html')
if __name__ == '__main__':
    app.run(debug=True)
