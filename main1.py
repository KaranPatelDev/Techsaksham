# import streamlit as st
# import cv2
# import os
# import csv
# import numpy as np
# from PIL import Image
# import pandas as pd
# import datetime
# import time
# import openpyxl  # To save attendance in Excel format

# # -------------------------------------------------- Functions -------------------------------------------------- #

# def assure_path_exists(path):
#     if not os.path.exists(path):
#         os.makedirs(path)

# def check_haarcascadefile():
#     if not os.path.isfile("haarcascade_frontalface_default.xml"):
#         st.error('Haarcascade file is missing. Please contact support.')
#         return False
#     return True

# def save_pass():
#     assure_path_exists("TrainingImageLabel/")
#     if os.path.isfile("TrainingImageLabel/psd.txt"):
#         with open("TrainingImageLabel/psd.txt", "r") as tf:
#             key = tf.read()
#     else:
#         st.warning("Password file not found! Register a new password.")
#         return

#     op = old_pass
#     newp = new_pass
#     confirm_newp = confirm_pass

#     if op == key:
#         if newp == confirm_newp:
#             with open("TrainingImageLabel/psd.txt", "w") as tf:
#                 tf.write(newp)
#             st.success("Password changed successfully!")
#         else:
#             st.error("New password and confirmation do not match!")
#     else:
#         st.error("Incorrect old password!")

# def validate_inputs(student_id, student_name):
#     if not student_id.isdigit():
#         st.error("ID must be numeric!")
#         return False
#     if not student_name.replace(" ", "").isalpha():
#         st.error("Name must contain only alphabets!")
#         return False
#     return True

# def take_images(student_id, student_name):
#     if not validate_inputs(student_id, student_name):
#         return

#     check_haarcascadefile()
#     assure_path_exists("StudentDetails/")
#     assure_path_exists("TrainingImage/")

#     cam = cv2.VideoCapture(0)
#     detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
#     sample_count = 0

#     while True:
#         ret, frame = cam.read()
#         gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = detector.detectMultiScale(gray_frame, 1.3, 5)
#         for (x, y, w, h) in faces:
#             sample_count += 1
#             face_img = gray_frame[y:y + h, x:x + w]
#             cv2.imwrite(f"TrainingImage/{student_name}.{student_id}.{sample_count}.jpg", face_img)
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

#         cv2.imshow("Taking Images", frame)
#         if cv2.waitKey(100) & 0xFF == ord('q') or sample_count >= 100:
#             break

#     cam.release()
#     cv2.destroyAllWindows()

#     with open("StudentDetails/StudentDetails.csv", "a+", newline="") as csvfile:
#         writer = csv.writer(csvfile)
#         writer.writerow([student_id, student_name])

#     st.success(f"Images captured successfully for ID: {student_id}")

# def train_images():
#     check_haarcascadefile()
#     assure_path_exists("TrainingImageLabel/")
#     recognizer = cv2.face.LBPHFaceRecognizer_create()

#     try:
#         faces, ids = get_images_and_labels("TrainingImage")
#         recognizer.train(faces, np.array(ids))
#         recognizer.save("TrainingImageLabel/Trainer.yml")
#         st.success("Model trained successfully!")
#     except Exception as e:
#         st.error(f"Error during training: {e}")

# def get_images_and_labels(path):
#     image_paths = [os.path.join(path, img) for img in os.listdir(path)]
#     face_samples = []
#     ids = []

#     for img_path in image_paths:
#         gray_img = Image.open(img_path).convert("L")
#         face_np = np.array(gray_img, "uint8")
#         student_id = int(os.path.split(img_path)[-1].split(".")[1])
#         face_samples.append(face_np)
#         ids.append(student_id)

#     return face_samples, ids

# def track_images():
#     check_haarcascadefile()
#     recognizer = cv2.face.LBPHFaceRecognizer_create()
#     recognizer.read("TrainingImageLabel/Trainer.yml")
#     face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

#     cam = cv2.VideoCapture(0)
#     attendance = []
#     recognized_students = []

#     while True:
#         ret, frame = cam.read()
#         gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)
#         for (x, y, w, h) in faces:
#             id, confidence = recognizer.predict(gray_frame[y:y + h, x:x + w])
#             if confidence < 50:
#                 name = get_name_by_id(id)
#                 if name != "Unknown" and name not in recognized_students:
#                     recognized_students.append(name)
#                     attendance.append((id, name, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
#                 cv2.putText(frame, f"{name}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
#             else:
#                 cv2.putText(frame, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

#         # Display real-time attendance list in Streamlit
#         st.write(f"**Attendance Recorded So Far:**")
#         df = pd.DataFrame(attendance, columns=["ID", "Name", "Timestamp"])
#         st.dataframe(df)

#         cv2.imshow("Tracking Attendance", frame)
#         if cv2.waitKey(1) == ord('q') or len(attendance) > 10:  # Close after 10 attendees
#             break

#     cam.release()
#     cv2.destroyAllWindows()

#     save_attendance(attendance)

# def save_attendance(attendance):
#     date = datetime.datetime.now().strftime("%Y-%m-%d")
#     file_name = f"Attendance/Attendance_{date}.csv"
#     excel_file_name = f"Attendance/Attendance_{date}.xlsx"

#     # Save CSV
#     with open(file_name, "a+", newline="") as csvfile:
#         writer = csv.writer(csvfile)
#         writer.writerow(["ID", "Name", "Timestamp"])
#         writer.writerows(attendance)

#     # Save Excel
#     df = pd.DataFrame(attendance, columns=["ID", "Name", "Timestamp"])
#     df.to_excel(excel_file_name, index=False)

#     st.success(f"Attendance saved to {file_name} and {excel_file_name}")

# def get_name_by_id(student_id):
#     try:
#         with open("StudentDetails/StudentDetails.csv", "r") as csvfile:
#             reader = csv.reader(csvfile)
#             for row in reader:
#                 if len(row) > 0 and row[0] == str(student_id):
#                     return row[1]
#     except Exception as e:
#         st.error(f"Error reading student details: {e}")
#     return "Unknown"

# # ------------------------------------------------- Streamlit UI ------------------------------------------------- #

# st.title("Face Recognition Attendance System")

# # User inputs
# st.sidebar.header("Student Information")
# student_id = st.sidebar.text_input("Enter Student ID")
# student_name = st.sidebar.text_input("Enter Name")

# if st.sidebar.button("Capture Images"):
#     take_images(student_id, student_name)

# if st.sidebar.button("Train Model"):
#     train_images()

# if st.sidebar.button("Track Attendance"):
#     track_images()

# st.sidebar.header("Change Password (Optional)")
# old_pass = st.sidebar.text_input("Enter Old Password", type="password")
# new_pass = st.sidebar.text_input("Enter New Password", type="password")
# confirm_pass = st.sidebar.text_input("Confirm New Password", type="password")

# if st.sidebar.button("Save New Password"):
#     save_pass()




import streamlit as st
import cv2
import os
import csv
import numpy as np
from PIL import Image
import pandas as pd
import datetime
import time
import hashlib
import openpyxl  # To save attendance in Excel format
import logging

# -------------------------------------------------- Setup Logging -------------------------------------------------- #
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", handlers=[logging.StreamHandler(), logging.FileHandler("attendance_system.log")])

# -------------------------------------------------- Functions -------------------------------------------------- #

def assure_path_exists(path):
    """Ensure that the directory path exists"""
    if not os.path.exists(path):
        os.makedirs(path)
        logging.info(f"Created path: {path}")

def check_haarcascadefile():
    """Check if the haarcascade file exists"""
    if not os.path.isfile("haarcascade_frontalface_default.xml"):
        st.error('Haarcascade file is missing. Please contact support.')
        logging.error("Haarcascade file is missing.")
        return False
    return True

def hash_password(password):
    """Hash passwords securely"""
    return hashlib.sha256(password.encode()).hexdigest()

def save_pass():
    """Change the password securely"""
    assure_path_exists("TrainingImageLabel/")

    # Check if the password file exists
    if not os.path.isfile("TrainingImageLabel/psd.txt"):
        st.warning("Password file not found! Register a new password.")
        return

    with open("TrainingImageLabel/psd.txt", "r") as tf:
        stored_hash = tf.read()

    if hash_password(old_pass) == stored_hash:
        if new_pass == confirm_pass:
            with open("TrainingImageLabel/psd.txt", "w") as tf:
                tf.write(hash_password(new_pass))
            st.success("Password changed successfully!")
            logging.info("Password changed successfully.")
        else:
            st.error("New password and confirmation do not match!")
            logging.error("New password and confirmation do not match!")
    else:
        st.error("Incorrect old password!")
        logging.error("Incorrect old password.")

def validate_inputs(student_id, student_name):
    """Validate the student input"""
    if not student_id.isdigit():
        st.error("ID must be numeric!")
        logging.error(f"Invalid ID input: {student_id}")
        return False
    if not student_name.replace(" ", "").isalpha():
        st.error("Name must contain only alphabets!")
        logging.error(f"Invalid Name input: {student_name}")
        return False
    return True

def take_images(student_id, student_name):
    """Capture student's face images"""
    if not validate_inputs(student_id, student_name):
        return

    if not check_haarcascadefile():
        return

    assure_path_exists("StudentDetails/")
    assure_path_exists("TrainingImage/")

    cam = cv2.VideoCapture(0)
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    sample_count = 0

    stframe = st.empty()

    while True:
        ret, frame = cam.read()
        if not ret:
            st.error("Failed to access the camera.")
            break
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray_frame, 1.3, 5)

        for (x, y, w, h) in faces:
            sample_count += 1
            face_img = gray_frame[y:y + h, x:x + w]
            cv2.imwrite(f"TrainingImage/{student_name}.{student_id}.{sample_count}.jpg", face_img)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        stframe.image(frame, channels="BGR", caption="Capturing Images...")

        if sample_count >= 100:
            break

    cam.release()

    with open("StudentDetails/StudentDetails.csv", "a+", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([student_id, student_name])

    st.success(f"Images captured successfully for ID: {student_id}")
    logging.info(f"Captured images for student {student_id}, {student_name}")

def train_images():
    """Train the face recognition model"""
    if not check_haarcascadefile():
        return

    assure_path_exists("TrainingImageLabel/")
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    try:
        faces, ids = get_images_and_labels("TrainingImage")
        recognizer.train(faces, np.array(ids))
        recognizer.save("TrainingImageLabel/Trainer.yml")
        st.success("Model trained successfully!")
        logging.info("Model trained successfully.")
    except Exception as e:
        st.error(f"Error during training: {e}")
        logging.error(f"Error during training: {e}")

def get_images_and_labels(path):
    """Get images and corresponding labels for training"""
    image_paths = [os.path.join(path, img) for img in os.listdir(path)]
    face_samples = []
    ids = []

    for img_path in image_paths:
        gray_img = Image.open(img_path).convert("L")
        face_np = np.array(gray_img, "uint8")
        student_id = int(os.path.split(img_path)[-1].split(".")[1])
        face_samples.append(face_np)
        ids.append(student_id)

    return face_samples, ids

def track_images():
    """Track faces for attendance"""
    if not check_haarcascadefile():
        return

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("TrainingImageLabel/Trainer.yml")
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    cam = cv2.VideoCapture(0)
    attendance = []
    recognized_students = []
    stframe = st.empty()

    while True:
        ret, frame = cam.read()
        if not ret:
            st.error("Failed to access the camera.")
            break
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)

        for (x, y, w, h) in faces:
            id, confidence = recognizer.predict(gray_frame[y:y + h, x:x + w])
            if confidence < 50:
                name = get_name_by_id(id)
                if name != "Unknown" and name not in recognized_students:
                    recognized_students.append(name)
                    attendance.append((id, name, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                cv2.putText(frame, f"{name}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        stframe.image(frame, channels="BGR", caption="Tracking Attendance...")

        if len(attendance) >= 10:
            break

    cam.release()
    save_attendance(attendance)

def save_attendance(attendance):
    """Save attendance to CSV and Excel files"""
    date = datetime.datetime.now().strftime("%Y-%m-%d")
    file_name = f"Attendance/Attendance_{date}.csv"
    excel_file_name = f"Attendance/Attendance_{date}.xlsx"

    assure_path_exists("Attendance/")

    # Save CSV
    with open(file_name, "a+", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["ID", "Name", "Timestamp"])
        writer.writerows(attendance)

    # Save Excel
    df = pd.DataFrame(attendance, columns=["ID", "Name", "Timestamp"])
    df.to_excel(excel_file_name, index=False)

    st.success(f"Attendance saved to {file_name} and {excel_file_name}")
    logging.info(f"Attendance saved to {file_name} and {excel_file_name}")

def get_name_by_id(student_id):
    """Retrieve student's name by ID"""
    try:
        with open("StudentDetails/StudentDetails.csv", "r") as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if len(row) > 0 and row[0] == str(student_id):
                    return row[1]
    except Exception as e:
        st.error(f"Error reading student details: {e}")
        logging.error(f"Error reading student details: {e}")
    return "Unknown"

# ------------------------------------------------- Streamlit UI ------------------------------------------------- #

st.title("Face Recognition Attendance System")

# User inputs
st.sidebar.header("Student Information")
student_id = st.sidebar.text_input("Enter Student ID")
student_name = st.sidebar.text_input("Enter Name")

if st.sidebar.button("Capture Images"):
    take_images(student_id, student_name)

if st.sidebar.button("Train Model"):
    train_images()

if st.sidebar.button("Track Attendance"):
    track_images()

st.sidebar.header("Change Password (Optional)")
old_pass = st.sidebar.text_input("Enter Old Password", type="password")
new_pass = st.sidebar.text_input("Enter New Password", type="password")
confirm_pass = st.sidebar.text_input("Confirm New Password", type="password")

if st.sidebar.button("Save New Password"):
    save_pass()

