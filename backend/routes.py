from flask import Flask, request, jsonify, render_template
from backend.models import Attendance, FaceEnrollment
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload_face', methods=['POST'])
def upload_face():
    person_name = request.form.get('person_name')
    file = request.files.get('file')
    
    if not person_name or not file:
        return jsonify({'error': 'Missing name or file'}), 400
    
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)
    FaceEnrollment.enroll_face(person_name, file_path)
    return jsonify({'message': 'Face enrolled successfully'})

@app.route('/retrain', methods=['POST'])
def retrain_model():
    # Placeholder for retraining logic
    return jsonify({'message': 'Retraining initiated'})

@app.route('/get_attendance', methods=['GET'])
def get_attendance():
    records = Attendance.get_attendance_records()
    return jsonify(records)

@app.route('/mark_attendance', methods=['POST'])
def mark_attendance():
    data = request.get_json()
    person_name = data.get('person_name')
    status = data.get('status')
    
    if not person_name or not status:
        return jsonify({'error': 'Missing data'}), 400
    
    Attendance.mark_attendance(person_name, status)
    return jsonify({'message': 'Attendance marked successfully'})

if __name__ == '__main__':
    app.run(debug=True)
