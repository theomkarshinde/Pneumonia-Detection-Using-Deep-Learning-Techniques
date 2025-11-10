from pymongo import MongoClient , ReturnDocument
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, Response, session, send_file
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from PIL import Image
from datetime import datetime
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import io
import base64
from pytz import timezone
from datetime import datetime
import os

#Create WSGI Application app is instance of class Flask
app=Flask(__name__)
app.secret_key = "supersecretkey123"  # Required for flashing messages
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

MODEL_PATHS = {
    'cnn': 'modelgi.h5',
    'xgboost': 'xgboost_pneumonia_model.pkl',
    'svm': 'svm_pneumonia_model.pkl',
    'random_forest': 'random_forest_pneumonia_model.pkl',
    'rnn': 'rnn_model.h5'
}

# Ensure upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Connect to the MongoDB server
client = MongoClient('mongodb://localhost:27017/')

# Select database
db = client['pneumonia_detection']
counter_collection = db['counters']
patients = db['patients']
mediimg = db['medicalimage']
diagnosis = db['diagnosis']
doctorlogin = db['doctors']
adminlogin = db['admin']

# Function to generate the next alphanumeric ID
def get_next_alphanumeric_id(sequence_name):
    # Increment the sequence value and retrieve it
    result = counter_collection.find_one_and_update(
        {"_id": sequence_name},
        {"$inc": {"sequence_value": 1}},
        return_document=ReturnDocument.AFTER
    )
    if result:
        prefix = result["prefix"]
        sequence_value = result["sequence_value"]
        return f"{prefix}{sequence_value:03d}"  # Zero-padded to 3 digits
    else:
        raise ValueError(f"No counter found for sequence_name: {sequence_name}")

@app.route('/')
def home():
    return render_template('index.html')

# Flask-Login setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

# User model for Flask-Login
class User(UserMixin):
    def __init__(self, user_id, username, password_hash):
        self.id = user_id
        self.username = username
        self.password_hash = password_hash

# Admin model for Flask-Login
class Admin(UserMixin):
    def __init__(self, admin_id, username, password_hash):
        self.id = admin_id
        self.username = username
        self.password_hash = password_hash

# Admin login route
@app.route('/admin_login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        session['adminname'] = request.form['username']
        adminpassword = request.form['password']
        adminname = session.get('adminname')
        
        # Hardcoded admin credentials for demonstration
        admin_data = adminlogin.find_one({"username":adminname})
        if admin_data and check_password_hash(admin_data["password"], adminpassword):
            admin = Admin(admin_data["_id"], admin_data["username"], admin_data["password"])
            login_user(admin)  
            flash("Admin login successful!", "success")
            print(adminname)
            print(current_user.is_authenticated)  # Debugging: Should print True if logged in
            print(current_user) 
            return redirect(url_for('admin_dashboard'))
        
        flash("Invalid username or password.", "danger")
    
    return render_template('admin_login.html')

# Admin dashboard route
@app.route('/admin_dashboard')
#@login_required
def admin_dashboard():
    selected_model = session.get('selected_model', 'cnn')  # Default to CNN
    #print(selected_model,"dv")
    all_doctors = list(doctorlogin.find({}))
    ist = timezone('Asia/Kolkata')
    # Convert last_login datetime to a string format for each doctor
    for doctor in all_doctors:
        last_login = doctor.get("last_login")
        if last_login and last_login != "None":
            last_login_ist = last_login.replace(tzinfo=timezone('UTC')).astimezone(ist)
            doctor["last_login_str"] = last_login_ist.strftime('%Y-%m-%d %H:%M:%S')
        else:
            doctor["last_login_str"] = "Never Logged In"
    return render_template('admin_dashboard.html', all_doctors=all_doctors,selected_model=selected_model)

# Admin view doctor activities route
@app.route('/doctor_activities/<doctor_id>')
#@login_required
def doctor_activities(doctor_id):
    adminname = session.get('adminname')
    
    doctor_activities = list(patients.find({"doctor_id": doctor_id}))
    # Fetch the diagnosis details for each patient
    for patient in doctor_activities:
        # Fetch medical image data instead of diagnosis
        diagnosiss =  mediimg.find_one({'patient_id': patient.get('patient_id')})
        if diagnosiss:
            patient['diagnosis_result'] = diagnosiss.get('diagnosis_result', 'Not Available') if diagnosiss else 'Not Available'
    return render_template('doctor_activities.html', doctor_activities=doctor_activities, doctor_id=doctor_id)

# Admin access Jupyter notebook route
@app.route('/set_model', methods=['POST'])
#@login_required
def set_model():
    selected_model = request.form.get('model', 'cnn')  # Get the selected model from the form
    session['selected_model'] = selected_model  # Save it to session
    flash(f'Model "{selected_model}" selected successfully!', 'success')  # Optional feedback
    if 'adminname' in session:
        return redirect(url_for('admin_dashboard'))
    elif 'logname' in session:
        return redirect(url_for('dashboard'))
    else:
        return redirect(url_for('home'))  # fallback

@login_manager.user_loader
def load_user(user_id):
    # Fetch the user from the database by ID
    user_data = doctorlogin.find_one({"_id": user_id})
    if user_data:
        return User(user_id=user_data["_id"], username=user_data["name"], password_hash=user_data["password"])
    return None 

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        logname = request.form['logusername']
        session['logname'] = logname
        logpassword = request.form['logpassword']
        
        # Find user in database
        user_data = doctorlogin.find_one({"doctor_id": logname})
        if user_data and check_password_hash(user_data["password"], logpassword):
            user = User(user_data["_id"], user_data["doctor_id"], user_data["password"])
            login_user(user)
            doctorlogin.update_one(
                {"_id": user_data["_id"]}, 
                {"$set": {"last_login": datetime.utcnow()}}
            )
            flash("Login successful!", "success")
            return redirect(url_for('dashboard'))
        
        flash("Invalid username or password.", "danger")
    
    return render_template('login.html')

@app.route('/doctor_forgot_password', methods=['GET', 'POST'])
def doctor_forgot_password():
    if request.method == 'POST':
        step = request.form.get("step")

        if step == "verify":
            doctor_id = request.form['doctor_id']
            name = request.form['name']
            email = request.form['email']

            doctor = doctorlogin.find_one({'doctor_id': doctor_id, 'name': name, 'email': email})
            if doctor:
                return render_template('doctor_reset_password.html', doctor_id=doctor_id)
            else:
                flash("Verification failed. Please check your details.", "danger")
        
        elif step == "reset":
            doctor_id = request.form['doctor_id']
            new_password = request.form['new_password']
            hashed_password = generate_password_hash(new_password)
            doctorlogin.update_one({'doctor_id': doctor_id}, {'$set': {'password': hashed_password}})
            flash("Password reset successfully.", "success")
            return redirect(url_for('login'))

    return render_template('doctor_forgot_password.html')

@app.route('/dashboard')
#@login_required
def dashboard():
    logname = session.get('logname')
    udata = doctorlogin.find_one({"doctor_id": logname})
    doctorname = udata["name"]
    session['doctorname'] = doctorname
    return render_template('dashboard.html')
    
# 🔹 Route to Serve Images (For `<img src>` in HTML)
@app.route("/image/<patient_id>")
def serve_image(patient_id):
    """Retrieve the binary image from MongoDB and return it as a response."""
    image_data = mediimg.find_one({"patient_id": patient_id})

    if image_data and "image_url" in image_data:
        return Response(image_data["image_url"], mimetype="image/jpeg")  # Adjust mimetype if needed

    return "Image not found", 404

#code to see patient history to the doctor
@app.route('/patient_history')
def patient_history():
    #doctor_id = current_user.id 
    logname = session.get('logname')
    doctor_id = logname  # Assuming the doctor's ID is stored in the username field
    patient_history = list(patients.find({"doctor_id": doctor_id}))
    # Fetch medical images and diagnosis results for each patient
    for patient in patient_history:
        medical_image_data = mediimg.find_one({"patient_id": patient["patient_id"]})
        print(type(medical_image_data["image_url"]))  # Should be <class 'bytes'>

        if medical_image_data and "image_url" in medical_image_data:
            try:
                # Important: ensure value is of type bytes before base64 encode
                image_bytes = medical_image_data["image_url"]
                if isinstance(image_bytes, bytes):
                    base64_image = base64.b64encode(image_bytes).decode('utf-8')
                    patient["image_url"] = f"data:image/jpeg;base64,{base64_image}"
                else:
                    print(f"Invalid image data type for patient {patient['patient_id']}")
                    patient["image_url"] = None
            except Exception as e:
                print(f"Error encoding image for patient {patient['patient_id']}: {e}")
                patient["image_url"] = None

            patient["diagnosis_result"] = medical_image_data.get("diagnosis_result", "No diagnosis available")
        else:
            patient["image_url"] = None
            patient["diagnosis_result"] = "No diagnosis available"

    return render_template('patient_history.html', patient_history=patient_history)

@app.route('/xray_upload', methods=['GET', 'POST'])
def xray_upload():
    return render_template('xray_upload.html')

@app.route('/upload_image', methods=['POST','GET'])
def upload_image():
    print(request.files)
    selected_model = session.get('selected_model')  # Default model
    print(selected_model,"hdbs")

    name = request.form['name']
    age = request.form['age']
    gender = request.form['gender']
    email = request.form['email']
    phone = request.form['phone']
    doctor_id = request.form['doctor_id']
    file = request.files['file-input']

    def load_model1():
        selected_model = session.get('selected_model', 'cnn')  # Default model
        print(selected_model,"hdbs")
        model_path = MODEL_PATHS.get(selected_model, MODEL_PATHS['cnn'])
    
        # Load the actual model (you need to implement this part based on model type)
        if model_path.endswith('.h5'):
            from tensorflow.keras.models import load_model
            model = load_model(model_path)
            print('cnn')
        elif model_path.endswith('.pkl'):
            import pickle
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
                print('xgboost')
        else:
            model = load_model('modelgi.h5')
        return model
    
    img_size = 150

    image = Image.open(file)
    binary_image = file.read()
    file.stream.seek(0)


    # Process the image
    image = Image.open(file.stream)  # Open image from file stream
    image = image.convert('L')  # Convert image to grayscale
    image = image.resize((150, 150))  # Resize image to match model input size
    image = np.array(image)  # Convert to numpy array
    image = image / 255.0  # Normalize the image
    image = image.reshape(-1, 150, 150, 1)  # Reshape image to fit model input
    
    # Predict the class (or output) using the trained model
    model = load_model1()
    print(model,"bhvjhbdcvh")
    modelvalidation = load_model('chest_image_validation.h5')
    
    file.stream.seek(0)
    
    imagevalidation = Image.open(file.stream)

    # Convert grayscale images to RGB
    if imagevalidation.mode != "RGB":
        imagevalidation = imagevalidation.convert("RGB")
        
    # Preprocess the image for model prediction
    imagevalidation = imagevalidation.resize((128, 128))  # Resize to match model input size
    image_array = tf.keras.utils.img_to_array(imagevalidation) / 255.0  # Normalize
    image_array = tf.expand_dims(image_array, axis=0)  # Add batch dimension
        
    # Predict using the loaded model
    predictionvalidation = modelvalidation.predict(image_array)[0][0]
    resultvalidation = "Chest X-ray" if predictionvalidation > 0.5 else "Non-Chest X-ray"
         
    
    if resultvalidation == "Chest X-ray":
        prediction = model.predict(image)
        print("prediction is:",prediction[0])
    
        # Determine the result
        if prediction[0] > 0.5:
            result1='Normal'
        else:
            result1='Pneumonia'

        next_patient_id = get_next_alphanumeric_id("patient_id")
        next_diagnosis_id = get_next_alphanumeric_id("diagnosis_id")
        
        new_patient = {
        "patient_id": next_patient_id,
        "name": name,
        "age": age,
        "gender": gender,
        "doctor_id": doctor_id,  # Referencing the doctor who uploaded the image
        "email": email,
        "phone":phone,
        "diagnosis_id": next_diagnosis_id  
        }

        # Insert patient into the collection
        patients.insert_one(new_patient)
    
        new_medi_img = {
        "doctor_id" : doctor_id,
        "patient_id": next_patient_id,
        "image_url" : binary_image,
        "upload_date": datetime.utcnow(),
        "diagnosis_result": result1
        }
    
        # Insert diagnosis result
        mediimg.insert_one(new_medi_img)
        
        return render_template('prediction_result.html', result=result1,probability=round(float(prediction[0]) * 100, 2))
    
    else:
        flash("Not chest X-ray image.")  
        
    return render_template('xray_upload.html')

@app.route('/logout')
#@login_required
def logout():
    session.clear()  # Logs out user by clearing session
    logout_user()
    flash("You have been logged out.", "info")
    return redirect(url_for('home'))

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    from_admin = request.args.get('from_admin') == 'true'
    
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        demail = request.form['demail']
        dphone = request.form['dphone']
        dspecial = request.form['dspecial']
        dreg = request.form['dreg']

        # Check if the user already exists
        if doctorlogin.find_one({"name": username}):
            flash("Username already exists!", "danger")
            return redirect(url_for('register'))
        
        # Save the new user
        password_hash = generate_password_hash(password)
        next_doctor_id = get_next_alphanumeric_id("doctor_id")
        doctor_id = next_doctor_id
        doctorlogin.insert_one({
                                "doctor_id": next_doctor_id, 
                                "name": username,
                                "email": demail,
                                "password": password_hash,
                                "phone": dphone,
                                "specialization": dspecial,
                                "registration_date": dreg,
                                "last_login":"None"
        })
        flash(f"Doctor registered successfully! Your Doctor ID is {doctor_id}", "success")
        # Redirect based on who registered
        if from_admin:
            return redirect(url_for('admin_dashboard'))
        else:
            return redirect(url_for('login'))
    
    return render_template('register.html')

if __name__ == '__main__':
    app.run(debug=True)
