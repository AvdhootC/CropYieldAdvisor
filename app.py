from flask import Flask, render_template, request, send_file, redirect, url_for, session
import joblib
import pandas as pd
import os
from dotenv import load_dotenv
import google.generativeai as genai
import pymysql
# Load environment variables
load_dotenv()
# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.0-flash")
# Set up the Flask app
app = Flask(__name__, static_url_path='', static_folder='static')
app.secret_key = 'your_secret_key' # Required for session management
# MySQL DB Config
db = pymysql.connect(
host="localhost", user="root", password="12345678", # replace with your MySQL root password
database="crop_yield_predictor"
)
# Load ML model and encoders
ml_model = joblib.load('xgb_crop_yield_model.pkl')
encoders = joblib.load('label_encoders.pkl')

# Store the last explanation (for download route)
last_explanation = ""
last_prediction = 0.0
last_inputs = {}
@app.route('/')
def signup():
return render_template('signup.html')
@app.route('/signup', methods=['POST'])
def handle_signup():
username = request.form['username']
password = request.form['password']
cursor = db.cursor()
# Check if user already exists
cursor.execute("SELECT * FROM users WHERE username=%s", (username,))
user = cursor.fetchone()
if user:
return "User already exists. <a href='/login'>Login here</a>" # Insert new user with username and password
cursor.execute("INSERT INTO users (username, password) VALUES (%s, %s)", (username, password))
db.commit()
return redirect(url_for('login'))
@app.route('/login')
def login():
return render_template('login.html')
@app.route('/login', methods=['POST'])
def handle_login():
username = request.form['username']
password = request.form['password']
cursor = db.cursor()
cursor.execute("SELECT * FROM users WHERE username=%s AND password=%s", (username, password))
user = cursor.fetchone()
if user:
session['username'] = username

return redirect(url_for('index'))
return "Invalid credentials. <a href='/login'>Try again</a>" @app.route('/index')
def index():
# Check if the user is logged in by checking the session
if 'username' in session:
return render_template('index.html', username=session['username'])
else:
return redirect(url_for('login'))
@app.route('/predict', methods=['POST'])
def predict():
try:
# Get language preference
language = request.form.get('language', 'English')
# Collect form data
data = {
'Region': request.form['Region'],
'Soil_Type': request.form['Soil_Type'],
'Crop': request.form['Crop'],
'Weather_Condition': request.form['Weather_Condition'],
'Fertilizer_Used': int(request.form['Fertilizer_Used']),
'Irrigation_Used': int(request.form['Irrigation_Used']),
'Rainfall_mm': float(request.form['Rainfall_mm']),
'Temperature_Celsius': float(request.form['Temperature_Celsius']),
'Days_to_Harvest': int(request.form['Days_to_Harvest']), }
# Encode categorical variables
for col in ['Region', 'Soil_Type', 'Crop', 'Weather_Condition']:
encoder = encoders[col]
data[col] = encoder.transform([data[col]])[0]
# Create DataFrame and match model input
input_df = pd.DataFrame([data])
input_df = input_df[ml_model.get_booster().feature_names]
# Predict yield
prediction = round(ml_model.predict(input_df)[0], 2)
# Store the prediction in the database
if 'username' in session:
username = session['username']

cursor = db.cursor()
# Insert prediction into the predictions table
cursor.execute("""
INSERT INTO crop_yield_predictions (
# Prepare input summary for GenAI
original_inputs = {
'Region': request.form['Region'],
'Soil Type': request.form['Soil_Type'],
'Crop': request.form['Crop'],
'Weather Condition': request.form['Weather_Condition'],
'Fertilizer Used (kg/ha)': request.form['Fertilizer_Used'],
'Irrigation Used (Litre/ha)': request.form['Irrigation_Used'],
'Rainfall (mm)': request.form['Rainfall_mm'],
'Temperature (°C)': request.form['Temperature_Celsius'],
'Days to Harvest': request.form['Days_to_Harvest']
}
# Generate explanation for the user in the selected language
prompt = f""" You are an expert agricultural advisor assisting rural farmers. Provide a detailed yet simple explanation **strictly in {language} only** based on the
following information:
{original_inputs}
The predicted crop yield is {prediction} quintals per hectare. � Format your response using the structure below. All section headers, bullet points, and explanations **must be fully translated into {language}**. Do NOT include any
English words, phrases, or symbols. � **इनपुट सारांश** (❌ This must be translated to {language})
- Clearly list each input with a short and simple explanation (✅ Use farmer-friendly
terms)
� **पूरावनुुान उपज** (❌Translate this too)
- Mention the predicted yield and explain whether it is low, average, or good in simple
terms
✅ **का अचा हु** (❌ Translate this heading too)
- Bullet points of positive or favorable conditions
⚠ **का सुधार किया जा सिता है
** (❌Translate this as well)
- Bullet points listing problems or areas that need improvement

� **करशेषज िी सलाह** (❌ Translate this heading too)
- 2 to 3 practical and easy-to-understand tips to improve future yield
� Use only: - Fully translated section headers in {language}
- Simple bullet points (• or -)
- Friendly emojis
- Very basic and relatable language for farmers
❌ Final response must be written entirely in {language}. Avoid mixing languages. English is **not allowed** anywhere in the response. """ # Generate GenAI response
gemini_response = model.generate_content(prompt)
explanation = gemini_response.text
return render_template('index.html', prediction=prediction, explanation=explanation, language=language)
except Exception as e:
return f"❌ Error occurred: {e}"
if __name__ == '__main__':
app.run(debug=True)
