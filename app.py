from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import cv2
import numpy as np
import os
from inference_sdk import InferenceHTTPClient
import random
import supabase

url="https://qdlfpetmbyuudwgchluh.supabase.co"
key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InFkbGZwZXRtYnl1dWR3Z2NobHVoIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDM4NzQ1ODksImV4cCI6MjA1OTQ1MDU4OX0.U2AtFBDjVlsYSmXw9LgjrHP_T8sfy3aMnPSfiUfiHiM"
slots = []
Client = supabase.create_client(url, key)
table_name = "parking_slots"

try:
    response = Client.table(table_name).select("*").execute()

    # API Response structure has 'data' and potentially 'error' keys
    if hasattr(response, 'data') and response.data:
        print(f"Successfully retrieved {len(response.data)} rows:")
        for row in response.data:
            if row['is_free'] == 'true':
                slots.append(row['slot_label'])
    # supabase-py might raise an exception on error, but good to check response too
    elif hasattr(response, 'error') and response.error:
         print(f"Error retrieving data: {response.error}")
    else:
         # Handle cases where data is empty list (valid response)
         print("No data found or unexpected response structure.")
         print("Response:", response) # Log the full response for debugging

except Exception as e:
    print(f"An error occurred during data retrieval: {e}")

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Roboflow client setup
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="gpm0fC7gaKhUgWSdHlfQ"
)

USER = {'username': 'admin', 'password': 'password'}

@app.route('/')
def home():
    return redirect(url_for('login'))  # Always go to login first

@app.route('/get-api', methods=['GET'])
def get_api():
    try:
        # Fetch data from the API
        response = slots
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == USER['username'] and password == USER['password']:
            session['username'] = username
            return redirect(url_for('dashboard'))
        else:
            error = 'Invalid username or password'
    return render_template('login.html', error=error)

@app.route('/dashboard')
def dashboard():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('dashboard.html', username=session['username'],data=slots)

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/capture-and-infer')
def capture_and_infer():
    cap = cv2.VideoCapture(1)  # Use Iriun webcam

    if not cap.isOpened():
        return jsonify({'error': 'Webcam not available'})

    ret, frame = cap.read()
    cap.release()

    if not ret:
        return jsonify({'error': 'Failed to capture frame'})

    image_path = 'static/captured.jpg'
    cv2.imwrite(image_path, frame)

    # Inference
    try:
        result = CLIENT.infer(image_path, model_id="parking-lot-fmzu9/1")
        predictions = result['predictions']
        total_slots = 8
        slots = [False] * total_slots  # Default all vacant

        for pred in predictions:
            slot_index = int(pred['class']) if pred['class'].isdigit() else -1
            if 0 <= slot_index < total_slots:
                slots[slot_index] = True

        return jsonify({'slots': slots})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/parking-status')
def parking_status():
    return redirect('/capture-and-infer')

if __name__ == '__main__':
    app.run(debug=True)
