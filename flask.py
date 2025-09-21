from flask import Flask, request, jsonify, render_template, redirect, url_for
import sqlite3
import jwt
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta

app = Flask(__name__)
app.config['SECRET_KEY'] = "YOUR_SECRET_KEY"

# Initialize database
def init_db():
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        password TEXT
    )
    """)
    conn.commit()
    conn.close()

init_db()

@app.route("/")
def home():
    return render_template("login.html")

# Register
@app.route("/register", methods=["POST"])
def register():
    data = request.form
    username = data["username"]
    password = generate_password_hash(data["password"])
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO users(username,password) VALUES (?,?)", (username,password))
        conn.commit()
    except:
        return "❌ User already exists"
    finally:
        conn.close()
    return "✅ Registered successfully. Please login."

# Login
@app.route("/login", methods=["POST"])
def login():
    data = request.form
    username = data["username"]
    password = data["password"]

    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    cursor.execute("SELECT password FROM users WHERE username=?", (username,))
    row = cursor.fetchone()
    conn.close()

    if row and check_password_hash(row[0], password):
        token = jwt.encode(
            {"user": username, "exp": datetime.utcnow()+timedelta(hours=2)},
            app.config["SECRET_KEY"],
            algorithm="HS256"
        )
        # Redirect to Streamlit app with token
        return redirect(f"http://localhost:8501?token={token}")
    return "❌ Invalid credentials"
