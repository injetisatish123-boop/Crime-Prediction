import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template, redirect, url_for, flash, session
import pickle
import json
import os
import hashlib
from functools import wraps
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__)
app.secret_key = 'super_secret_matrix_key' # In production, use an environment variable

# Path to users JSON file
USERS_FILE = 'users.json'

# Helper functions for user management
def load_users():
    if not os.path.exists(USERS_FILE):
        return {}
    with open(USERS_FILE, 'r') as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {}

def save_users(users):
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=4)

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Decorator to protect routes
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_email' not in session:
            flash('Please log in to access this system.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Loading Models
decision = pickle.load(open('Decision.pkl', 'rb'))
Bagging = pickle.load(open('Bagging.pkl', 'rb'))

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        users = load_users()
        hashed_pw = hash_password(password)
        
        if email in users and users[email]['password'] == hashed_pw:
            session['user_email'] = email
            session['user_name'] = users[email]['name']
            return redirect(url_for('upload'))
        else:
            flash('Invalid credentials. Access Denied.', 'danger')
            return render_template('login.html', error="Invalid Credentials")
            
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')
        
        users = load_users()
        
        if email in users:
            flash('Email already registered. Try logging in.', 'info')
            return render_template('signup.html', error="Email already exists")
            
        users[email] = {
            'name': name,
            'password': hash_password(password)
        }
        save_users(users)
        flash('Account created successfully! Initialize session.', 'success')
        return redirect(url_for('login'))
        
    return render_template('signup.html')

@app.route('/logout')
def logout():
    session.pop('user_email', None)
    session.pop('user_name', None)
    flash('Session terminated.', 'info')
    return redirect(url_for('index'))

@app.route('/forgot-password')
def forgot_password():
    return render_template('forgot_password.html')

@app.route('/upload')
@login_required
def upload():
    return render_template('upload.html')

@app.route('/preview', methods=["POST"])
@login_required
def preview():
    if request.method == 'POST':
        dataset = request.files['datasetfile']
        df = pd.read_csv(dataset, encoding='unicode_escape')
        df.set_index('Id', inplace=True)
        return render_template("preview.html", df_view=df.head(100))

@app.route('/prediction', methods=['GET', 'POST'])
@login_required
def prediction():
    return render_template('prediction.html')

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    if request.method == 'POST':
        Year = request.form['Year']
        Month = request.form['Month']
        Day = request.form['Day']
        dayOfWeek = request.form['dayOfWeek']
        crimeAgainst = request.form['crimeAgainst'] 
        near_place = request.form['near_place']
        latitude = request.form['latitude']
        longitude = request.form['longitude']
        Model = request.form['Model']
        
        sample_data = [Year, Month, Day, dayOfWeek, crimeAgainst, near_place, latitude, longitude]
        int_feature = [float(i) for i in sample_data]
        ex1 = np.array(int_feature).reshape(1, -1)

        result_prediction = ""
        if Model == 'DecisionTreeClassifier':
            result_prediction = decision.predict(ex1)
        elif Model == 'BaggingClassifier':
            result_prediction = Bagging.predict(ex1)
        
        # Store result in session and redirect to dedicated result page
        session['prediction_result'] = str(result_prediction[0])
        session['prediction_model'] = Model
        session['prediction_inputs'] = {
            'Year': Year, 'Month': Month, 'Day': Day,
            'dayOfWeek': dayOfWeek, 'crimeAgainst': crimeAgainst,
            'near_place': near_place, 'latitude': latitude, 'longitude': longitude
        }
        return redirect(url_for('result'))

@app.route('/result')
@login_required
def result():
    prediction_text = session.pop('prediction_result', None)
    model = session.pop('prediction_model', None)
    inputs = session.pop('prediction_inputs', {})
    if not prediction_text:
        return redirect(url_for('prediction'))
    return render_template('result.html', prediction_text=prediction_text, model=model, inputs=inputs)

@app.route('/chart')
@login_required
def chart():
    return render_template('chart.html')

@app.route('/performance')
@login_required
def performance():
    # Per-class metrics for Decision Tree
    dt_report = {
        0: {'recall': 1.00, 'precision': 1.00, 'f1': 0.99},
        1: {'recall': 0.99, 'precision': 1.00, 'f1': 0.99},
    }
    dt_confusion = [[330, 3], [5, 327]]

    # Per-class metrics for Bagging Classifier
    bag_report = {
        0: {'recall': 0.97, 'precision': 1.00, 'f1': 0.98},
        1: {'recall': 1.00, 'precision': 0.97, 'f1': 0.98},
    }
    bag_confusion = [[325, 8], [0, 332]]

    labels = [0, 1]

    # Generate confusion matrix heatmap for Decision Tree
    fig1, ax1 = plt.subplots(figsize=(5, 4))
    sns.heatmap(dt_confusion, annot=True, fmt='d', cmap='coolwarm', ax=ax1,
                xticklabels=labels, yticklabels=labels)
    ax1.set_xlabel('x_predict')
    ax1.set_ylabel('y')
    ax1.set_title('')
    dt_img_path = os.path.join('static', 'images', 'dt_confusion.png')
    os.makedirs(os.path.join('static', 'images'), exist_ok=True)
    fig1.savefig(dt_img_path, bbox_inches='tight', dpi=100)
    plt.close(fig1)

    # Generate confusion matrix heatmap for Bagging Classifier
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    sns.heatmap(bag_confusion, annot=True, fmt='d', cmap='coolwarm', ax=ax2,
                xticklabels=labels, yticklabels=labels)
    ax2.set_xlabel('x_predict')
    ax2.set_ylabel('y')
    ax2.set_title('')
    bag_img_path = os.path.join('static', 'images', 'bag_confusion.png')
    fig2.savefig(bag_img_path, bbox_inches='tight', dpi=100)
    plt.close(fig2)

    return render_template('performance.html',
                           dt_report=dt_report,
                           bag_report=bag_report,
                           dt_img='images/dt_confusion.png',
                           bag_img='images/bag_confusion.png')

if __name__ == "__main__":
    app.run(debug=True, port=5001)
