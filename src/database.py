import sqlite3
import bcrypt
from datetime import datetime
import json
import os

class Database:
    def __init__(self, db_path='data/users.db'):
        # Ensure data directory exists
        os.makedirs('data', exist_ok=True)
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize all tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Prediction history table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                symptoms TEXT,
                predicted_disease TEXT,
                confidence REAL,
                top_predictions TEXT,
                prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # Appointments table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS appointments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                patient_name TEXT,
                patient_email TEXT,
                patient_phone TEXT,
                doctor_name TEXT,
                appointment_date TEXT,
                appointment_time TEXT,
                disease TEXT,
                status TEXT DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        conn.commit()
        conn.close()
        print(f"✅ Database initialized at {self.db_path}")
    
    def create_user(self, username, email, password):
        """Create a new user"""
        try:
            hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
                (username, email, hashed)
            )
            conn.commit()
            user_id = cursor.lastrowid
            conn.close()
            return user_id, None
        except sqlite3.IntegrityError as e:
            return None, str(e)
    
    def verify_user(self, username, password):
        """Verify user credentials"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id, username, email, password FROM users WHERE username = ?", (username,))
        user = cursor.fetchone()
        conn.close()
        
        if user and bcrypt.checkpw(password.encode('utf-8'), user[3]):
            return {'id': user[0], 'username': user[1], 'email': user[2]}
        return None
    
    def save_prediction(self, user_id, symptoms, predicted_disease, confidence, top_predictions):
        """Save prediction to history"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO predictions (user_id, symptoms, predicted_disease, confidence, top_predictions) VALUES (?, ?, ?, ?, ?)",
            (user_id, json.dumps(symptoms), predicted_disease, confidence, json.dumps(top_predictions))
        )
        conn.commit()
        conn.close()
    
    def get_prediction_history(self, user_id, limit=10):
        """Get user's prediction history"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT symptoms, predicted_disease, confidence, prediction_date FROM predictions WHERE user_id = ? ORDER BY prediction_date DESC LIMIT ?",
            (user_id, limit)
        )
        history = cursor.fetchall()
        conn.close()
        
        result = []
        for row in history:
            result.append({
                'symptoms': json.loads(row[0]),
                'disease': row[1],
                'confidence': row[2],
                'date': row[3]
            })
        return result
    
    def save_appointment(self, user_id, patient_name, patient_email, patient_phone, doctor_name, appointment_date, appointment_time, disease):
        """Save appointment booking"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO appointments (user_id, patient_name, patient_email, patient_phone, doctor_name, appointment_date, appointment_time, disease) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (user_id, patient_name, patient_email, patient_phone, doctor_name, appointment_date, appointment_time, disease)
        )
        conn.commit()
        appointment_id = cursor.lastrowid
        conn.close()
        return appointment_id
    
    def get_appointments(self, user_id):
        """Get user's appointments"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT patient_name, doctor_name, appointment_date, appointment_time, disease, status FROM appointments WHERE user_id = ? ORDER BY appointment_date DESC",
            (user_id,)
        )
        appointments = cursor.fetchall()
        conn.close()
        
        result = []
        for row in appointments:
            result.append({
                'patient_name': row[0],
                'doctor_name': row[1],
                'date': row[2],
                'time': row[3],
                'disease': row[4],
                'status': row[5]
            })
        return result