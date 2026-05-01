from flask import Flask, render_template, jsonify, request, redirect, url_for, session, flash
import heart_rate
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
import os
import re
import numpy as np
import cv2

app = Flask(__name__)
app.secret_key = os.urandom(24)

# =========================
# DATABASE SETUP
# =========================
def init_db():
    conn = sqlite3.connect('heartrate_users.db')
    c = conn.cursor()

    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                 username TEXT UNIQUE NOT NULL,
                 password TEXT NOT NULL,
                 email TEXT)''')

    c.execute('''CREATE TABLE IF NOT EXISTS heartrate_data
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                 user_id INTEGER NOT NULL,
                 bpm INTEGER NOT NULL,
                 timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                 FOREIGN KEY(user_id) REFERENCES users(id))''')

    conn.commit()
    conn.close()

init_db()


def get_db_connection():
    conn = sqlite3.connect('heartrate_users.db')
    conn.row_factory = sqlite3.Row
    return conn


# =========================
# ROUTES
# =========================
@app.route('/')
def index():
    if 'user_id' in session:
        conn = get_db_connection()

        readings = conn.execute('''
            SELECT bpm, timestamp 
            FROM heartrate_data 
            WHERE user_id = ? 
            ORDER BY timestamp DESC 
            LIMIT 5
        ''', (session['user_id'],)).fetchall()

        latest_reading = conn.execute('''
            SELECT bpm 
            FROM heartrate_data 
            WHERE user_id = ? 
            ORDER BY timestamp DESC 
            LIMIT 1
        ''', (session['user_id'],)).fetchone()

        conn.close()
        return render_template('index.html', readings=readings, latest_reading=latest_reading)

    return redirect(url_for('login'))


# =========================
# AUTH
# =========================
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn = get_db_connection()
        user = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
        conn.close()

        if user and check_password_hash(user['password'], password):
            session['user_id'] = user['id']
            session['username'] = user['username']
            flash('Logged in successfully!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password', 'danger')

    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']

        if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
            flash('Please enter a valid email address.', 'danger')
            return redirect(url_for('register'))

        conn = get_db_connection()
        try:
            conn.execute(
                'INSERT INTO users (username, password, email) VALUES (?, ?, ?)',
                (username, generate_password_hash(password), email)
            )
            conn.commit()
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))

        except sqlite3.IntegrityError:
            flash('Username already exists', 'danger')

        finally:
            conn.close()

    return render_template('register.html')


@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out', 'info')
    return redirect(url_for('login'))


# =========================
# MONITOR PAGE
# =========================
@app.route('/monitor')
def monitor():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('monitor.html')


# =========================
# 🔥 CORE API (IMPORTANT)
# =========================
@app.route('/process_frame', methods=['POST'])
def process_frame():
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401

    file = request.files.get('frame')

    if not file:
        return jsonify({'error': 'No frame received'}), 400

    import numpy as np
    import cv2

    file_bytes = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    bpm = heart_rate.process_frame(frame)

    return jsonify({
        'bpm': bpm
    })

@app.route('/capture_reading', methods=['POST'])
def capture_reading():
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401

    bpm = request.json.get('bpm')

    if not bpm or bpm <= 0:
        return jsonify({'error': 'Invalid BPM'}), 400

    conn = get_db_connection()

    conn.execute(
        'INSERT INTO heartrate_data (user_id, bpm) VALUES (?, ?)',
        (session['user_id'], bpm)
    )
    conn.commit()

    readings = conn.execute('''
        SELECT bpm, timestamp 
        FROM heartrate_data 
        WHERE user_id = ? 
        ORDER BY timestamp DESC 
        LIMIT 5
    ''', (session['user_id'],)).fetchall()

    conn.close()

    return jsonify({
        'message': 'Saved',
        'readings': [dict(r) for r in readings]
    })


# =========================
# CLEAR HISTORY
# =========================
@app.route('/clear_readings', methods=['POST'])
def clear_readings():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    conn = get_db_connection()
    conn.execute('DELETE FROM heartrate_data WHERE user_id = ?', (session['user_id'],))
    conn.commit()
    conn.close()

    flash('Your reading history has been cleared.', 'info')
    return redirect(url_for('index'))


# =========================
# STATIC PAGES
# =========================
@app.route('/about')
def about():
    return render_template('about.html')


# =========================
# RUN APP
# =========================
if __name__ == "__main__":
    app.run()