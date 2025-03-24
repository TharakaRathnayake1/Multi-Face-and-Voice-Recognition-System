import os
import numpy as np
import torch
import torch.nn.functional as F
import pymysql
import jwt
import datetime
from facenet_pytorch import InceptionResnetV1, MTCNN  # Face recognition model (Adhikari et al., 2019)
from PIL import Image
from flask import Flask, request, jsonify, render_template, session, redirect, url_for
from pymongo import MongoClient
from datetime import datetime
from bson.objectid import ObjectId
from flask import Flask, request, jsonify
from pymongo import MongoClient
from functools import wraps
from flask import Flask, render_template, request, redirect, url_for, flash, session
from werkzeug.security import generate_password_hash


# -------------------------------
# Configuration & Initialization
# -------------------------------
dataset_dir = "./datasets/faces"
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)
    print("Created folder:", dataset_dir)
else:
    print("Dataset folder exists:", dataset_dir)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

mongo_uri = "mongodb+srv://pasanmahee:fVgPys0uknMCuT8S@cluster0.zwasfmo.mongodb.net/facedetectionattendance?retryWrites=true&w=majority"
client = MongoClient(mongo_uri)
db = client.facedetectionattendance
attendance_collection = db.attendance

# # Database connection
# def get_db_connection():
#     connection = pymysql.connect(
#         host='localhost',
#         user='root',
#         password='Tharuicbt01',
#         database='attendance_db',
#         cursorclass=pymysql.cursors.DictCursor
#     )
#     return connection

# -------------------------------
# 1. Load Face Recognition Models
# -------------------------------
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
mtcnn = MTCNN(image_size=160, margin=20, keep_all=False, device=device)

# -------------------------------
# 2. Enrollment Function
# -------------------------------
def enroll_faces(dataset_dir):
    """
    Enrolls faces by computing an average embedding per person from images in each subfolder.
    """
    enrolled = {}
    for person_name in os.listdir(dataset_dir):
        person_path = os.path.join(dataset_dir, person_name)
        if not os.path.isdir(person_path):
            continue

        embeddings_list = []
        for img_file in os.listdir(person_path):
            if not img_file.lower().endswith(('.jpg','.jpeg','.png')):
                continue
            img_path = os.path.join(person_path, img_file)
            img = Image.open(img_path).convert('RGB')
            face_tensor = mtcnn(img)
            if face_tensor is None:
                continue
            face_tensor = face_tensor.unsqueeze(0).to(device)
            with torch.no_grad():
                embedding = resnet(face_tensor)
            embeddings_list.append(embedding.cpu().numpy())

        if embeddings_list:
            avg_emb = np.mean(embeddings_list, axis=0)
            avg_emb_torch = torch.from_numpy(avg_emb).to(device)
            enrolled[person_name] = avg_emb_torch
            print(f"Enrolled {person_name} with {len(embeddings_list)} images.")
        else:
            print(f"No valid faces found for {person_name}, skipping enrollment.")
    return enrolled

enrolled_faces = enroll_faces(dataset_dir)


# -------------------------------
# 3. Identification Function
# -------------------------------
def identify_face(face_embedding, enrolled_dict, threshold=0.7):
    """
    Identifies a face by comparing the embedding with each enrolled person's average embedding using cosine similarity.
    """
    best_name = "unknown"
    best_score = -1.0
    for name, ref_emb in enrolled_dict.items():
        similarity = F.cosine_similarity(face_embedding, ref_emb, dim=1)
        sim_val = similarity.item()
        if sim_val > best_score:
            best_score = sim_val
            best_name = name

    if best_score < threshold:
        return "unknown", best_score
    else:
        return best_name, best_score

# -------------------------------
# 4. Flask App & API Endpoints
# -------------------------------
app = Flask(__name__)
app.secret_key = "your_secret_key_here"  # For session management (Flask, 2023)


# # JWT Token

# @app.route('/login/teacher', methods=['GET', 'POST'])
# def login_teacher():
#     error = None
#     if request.method == 'POST':
#         username = request.form.get('username')
#         password = request.form.get('password')
#         if username == TEACHER_USERNAME and password == TEACHER_PASSWORD:
#             # Generate a JWT token for the teacher
#             token = jwt.encode({
#                 'role': 'teacher', 
#                 'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=1)  # Token expiration time (1 hour)
#             }, SECRET_KEY, algorithm='HS256') # type: ignore
#             session['token'] = token  # Store the token in session
#             return redirect(url_for('dashboard_teacher'))
#         else:
#             error = "Invalid teacher credentials. Please try again."
#     return render_template('login_teacher.html', error=error)

# # Token Validation Decorator
# def token_required(f):
#     @wraps(f)
#     def decorated_function(*args, **kwargs):
#         token = None
#         # Check if the token is passed in the Authorization header
#         if 'Authorization' in request.headers:
#             token = request.headers['Authorization'].split(" ")[1]  # Format: Bearer <Token>
        
#         if not token:
#             return jsonify({'error': 'Token is missing!'}), 403  # Forbidden
        
#         try:
#             # Decode the token
#             data = jwt.decode(token, SECRET_KEY, algorithms=['HS256']) # type: ignore
#             current_user_role = data['role']
#         except jwt.ExpiredSignatureError:
#             return jsonify({'error': 'Token has expired!'}), 401  # Unauthorized
#         except jwt.InvalidTokenError:
#             return jsonify({'error': 'Invalid token!'}), 401  # Unauthorized
        
#         # Attach the role to the request context for later use
#         request.user_role = current_user_role
#         return f(*args, **kwargs)
    
#     return decorated_function

# # Apply Token Validation to Protected Routes
# @app.route('/identify', methods=['POST'])
# @token_required
# def identify_route():
#     # Only allow students to mark attendance via face recognition
#     if request.user_role != 'student':
#         return jsonify({'error': 'Attendance marking is only available for students.'}), 401
    
#     # The rest of the attendance marking code...

# # Handling Token in Other Routes
# @app.route('/upload_face', methods=['POST'])
# @token_required
# def upload_face():
#     # Only teachers can upload new face images
#     if request.user_role != 'teacher':
#         return jsonify({'error': 'Unauthorized: Only teachers can upload face images.'}), 401
    
#     # Image upload logic...

# # Error Handling
# @app.route('/identify', methods=['POST'])
# @token_required
# def identify_route():
#     if request.user_role != 'student':
#         return jsonify({'error': 'Attendance marking is only available for students.'}), 401

#     if 'file' not in request.files:
#         return jsonify({'error': 'No file part in the request'}), 400
#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({'error': 'No file selected'}), 400

#     try:
#         img = Image.open(file.stream).convert('RGB')
#     except Exception as e:
#         return jsonify({'error': 'Invalid image file'}), 400

#     # Continue the face recognition process...
# # Logout & Token Expiration
# @app.route('/logout')
# def logout():
#     session.clear()  # Clears the session, effectively logging out the user
#     return redirect(url_for('index'))

# # Example of Handling Expired Tokens
# {
#   "error": "Token has expired!"
# }

# Define credentials for each role (for demonstration purposes) (Flask, 2023)
TEACHER_USERNAME = "teacher"
TEACHER_PASSWORD = "teacherpass"
TEACHER_USERNAME = "teacher2"
TEACHER_PASSWORD = "teacherpass2"
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "adminpass"
ADMIN_USERNAME = "admin2"
ADMIN_PASSWORD = "admin2"
STUDENT_USERNAME = "student"
STUDENT_PASSWORD = "studentpass"
STUDENT_USERNAME = "student2"
STUDENT_PASSWORD = "student2"


# Public homepage route
@app.route('/')
def index():
    return render_template('index.html')

# ✅ Define `users_collection`
users_collection = db.users  # Ensure the collection name is 'users'

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        role = request.form['role']  # Get role from form

        if password != confirm_password:
            flash("Passwords do not match!", "danger")
            return redirect(url_for('register'))  # Updated route name

        # Hash password before storing
        hashed_password = generate_password_hash(password)

        # Insert into MongoDB
        users_collection.insert_one({
            "username": username,
            "email": email,
            "password": hashed_password,
            "role": role
        })

        flash("User registered successfully!", "success")
        return redirect(url_for('register'))  # Updated route name

    return render_template('register.html')  # No need to change the template name

# -------------------------------
# Identification Endpoint (Student Attendance)
# -------------------------------
@app.route('/identify', methods=['POST'])
def identify_route():
    # Only allow students to mark attendance via face recognition
    if session.get('role') != 'student':
        return jsonify({'error': 'Attendance marking is only available for students.'}), 401

    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    try:
        img = Image.open(file.stream).convert('RGB')
    except Exception as e:
        return jsonify({'error': 'Invalid image file'}), 400

    face_crop = mtcnn(img)
    if face_crop is None:
        return jsonify({'error': 'No face detected in the image'}), 400
    face_crop = face_crop.unsqueeze(0).to(device)
    with torch.no_grad():
        test_emb = resnet(face_crop)

    name, score = identify_face(test_emb, enrolled_faces, threshold=0.7)
    feedback = "Attendance not marked."
    if name != "unknown":
        today = datetime.today().strftime("%Y-%m-%d")
        attendance_collection.update_one(
            {"person_name": name, "date": today},
            {"$set": {"person_name": name, "date": today, "timestamp": datetime.utcnow()}},
            upsert=True
        )
        feedback = f"Attendance recorded for {name}."
    return jsonify({'detected_face': name, 'similarity_score': score, 'feedback': feedback})

# -------------------------------
# Enrollment & Training Endpoints (Teacher Functions)
# -------------------------------
@app.route('/upload_face', methods=['POST'])
def upload_face():
    # Only teachers can upload new face images
    if session.get('role') != 'teacher':
        return jsonify({'error': 'Unauthorized: Only teachers can upload face images.'}), 401

    person_name = request.form.get("person_name")
    if not person_name:
        return jsonify({'error': 'Missing person_name field'}), 400
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        img = Image.open(file.stream).convert('RGB')
    except Exception as e:
        return jsonify({'error': 'Invalid image file'}), 400

    person_dir = os.path.join(dataset_dir, person_name)
    os.makedirs(person_dir, exist_ok=True)

    existing_files = [f for f in os.listdir(person_dir) if f.startswith("img") and f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    next_index = len(existing_files) + 1
    new_filename = f"img{next_index}.jpg"
    save_path = os.path.join(person_dir, new_filename)
    try:
        img.save(save_path, format="JPEG")
    except Exception as e:
        return jsonify({'error': f'Failed to save image: {str(e)}'}), 500

    return jsonify({'message': 'Image uploaded successfully', 'file_path': save_path})

@app.route('/voice_attendance', methods=['POST'])
def voice_attendance():
    data = request.json
    voice_input = data.get('voice_input', '').lower().strip()  # Normalize input

    if voice_input == "present":
        return jsonify({"success": True, "confidence_score": 0.95})  # Example score
    else:
        return jsonify({"success": False, "message": "Invalid response."})

@app.route('/retrain', methods=['POST'])
def retrain():
    # Only teachers can trigger retraining of enrollment data
    if session.get('role') != 'teacher':
        return jsonify({'error': 'Unauthorized: Only teachers can retrain enrollment.'}), 401

    global enrolled_faces
    enrolled_faces = enroll_faces(dataset_dir)
    return jsonify({'message': 'Enrollment updated', 'num_persons': len(enrolled_faces)})

@app.route('/get_attendance', methods=['GET'])
def get_attendance():
    records = attendance_collection.find({})
    
    # Convert the records to a list with the necessary fields
    attendance_list = [
        {
            "person_name": rec.get("person_name", ""),
            "date": rec.get("date", ""),
            "timestamp": rec.get("timestamp", ""),
            "status": rec.get("status", "")
        } for rec in records
    ]
    
    return jsonify(attendance_list)
# -------------------------------
# Login Endpoints for Different Roles
# -------------------------------
@app.route('/login/teacher', methods=['GET', 'POST'])
def login_teacher():
    error = None
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if username == TEACHER_USERNAME and password == TEACHER_PASSWORD:
            session['logged_in'] = True
            session['role'] = 'teacher'
            return redirect(url_for('dashboard_teacher'))
        else:
            error = "Invalid teacher credentials. Please try again."
    return render_template('login_teacher.html', error=error)

@app.route('/login/admin', methods=['GET', 'POST'])
def login_admin():
    error = None
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session['logged_in'] = True
            session['role'] = 'admin'
            return redirect(url_for('dashboard_admin'))
        else:
            error = "Invalid admin credentials. Please try again."
    return render_template('login_admin.html', error=error)

@app.route('/login/student', methods=['GET', 'POST'])
def login_student():
    error = None
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if username == STUDENT_USERNAME and password == STUDENT_PASSWORD:
            session['logged_in'] = True
            session['role'] = 'student'
            return redirect(url_for('dashboard_student'))
        else:
            error = "Invalid student credentials. Please try again."
    return render_template('login_student.html', error=error)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

# -------------------------------
# Dashboard Endpoints for Different Roles
# -------------------------------
@app.route('/dashboard/teacher')
def dashboard_teacher():
    if session.get('role') != 'teacher':
        return redirect(url_for('login_teacher'))
    # Render teacher dashboard for enrollment and retraining functions
    return render_template('dashboard_teacher.html')

@app.route('/dashboard/admin')
def dashboard_admin():
    if session.get('role') != 'admin':
        return redirect(url_for('login_admin'))
    records = list(attendance_collection.find())
    return render_template('dashboard_admin.html', records=records)

@app.route('/dashboard/student')
def dashboard_student():
    if session.get('role') != 'student':
        return redirect(url_for('login_student'))
    # Render student dashboard for attendance functions
    return render_template('dashboard_student.html')

# -------------------------------
# Admin Attendance Edition Endpoint
# -------------------------------
@app.route("/update_attendance", methods=["POST"])
def update_attendance():
    data = request.json
    record_id = data.get("record_id")
    
    db.attendance.update_one(
        {"_id": record_id},
        {"$set": {
            "person_name": data["person_name"],
            "date": data["date"],
            "timestamp": data["timestamp"]
        }}
    )
    return jsonify({"message": "Record updated successfully!"})

# -------------------------------
# Admin Attendance Deletion Endpoint
# -------------------------------
@app.route('/delete_attendance', methods=['GET'])
def delete_attendance():
    if session.get('role') != 'admin':
        return redirect(url_for('login_admin'))
    record_id = request.args.get('record_id')
    if record_id:
        try:
            attendance_collection.delete_one({"_id": ObjectId(record_id)})
        except Exception as e:
            print("Error deleting record:", e)
    return redirect(url_for('dashboard_admin'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
