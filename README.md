# Heart Rate Monitoring System Using Computer Vision

## Overview

This project is a web-based heart rate monitoring system that uses a webcam to estimate a user's heart rate in real-time. It combines computer vision, signal processing, and a Flask-based web application to deliver live BPM readings along with user authentication and history tracking.

The system detects facial regions, extracts color signals, applies filtering techniques, and computes heart rate using frequency-domain and peak detection methods.

---

## Features

* Real-time heart rate estimation using webcam
* Face detection and region-of-interest tracking
* Signal processing using Butterworth filtering and FFT
* BPM calculation with smoothing and confidence estimation
* User authentication (login and registration)
* Secure password hashing
* SQLite database integration
* User-specific heart rate history tracking
* Live video streaming in browser
* Light level detection for measurement accuracy

---

## Project Structure

```id="6h9xqp"
project-root/
│
├── app.py
├── heart_rate.py
├── heartrate_users.db
│
├── templates/
│   ├── index.html
│   ├── login.html
│   ├── register.html
│   ├── monitor.html
│   ├── about.html
│
├── static/
│   ├── css/
│   ├── js/
│
├── deploy.prototxt
├── res10_300x300_ssd_iter_140000.caffemodel
│
└── README.md
```

---

## Core Components

### Flask Backend

Handles routing, authentication, session management, and database operations
Reference: 

### Heart Rate Processing Module

Implements signal processing pipeline including:

* Face detection (Haar Cascade)
* ROI extraction (forehead and cheek)
* Signal smoothing (Savitzky-Golay)
* Bandpass filtering (Butterworth)
* FFT and peak detection for BPM calculation
  Reference: 

### Database

* SQLite database for storing users and BPM readings
* Tracks last 5 readings per user

---

## How It Works

1. Webcam captures live video frames
2. Face is detected using Haar Cascade
3. Regions (forehead and cheek) are extracted
4. RGB signals are collected over time
5. Signal is filtered and processed
6. BPM is calculated using:

   * Frequency analysis (FFT)
   * Peak interval detection
7. Results are displayed in real-time and stored in database

---

## Technologies Used

* Python
* Flask
* OpenCV
* NumPy
* SciPy
* SQLite
* HTML / CSS / JavaScript

---

## How to Run

1. Install dependencies:

```id="c3mxwq"
pip install flask opencv-python numpy scipy
```

2. Ensure the following files are present:

* `deploy.prototxt`
* `res10_300x300_ssd_iter_140000.caffemodel`

3. Run the application:

```id="f4s8kp"
python app.py
```

4. Open browser:

```id="y2l9vd"
http://127.0.0.1:5000
```

---

## Limitations

* Accuracy depends heavily on lighting conditions
* Sensitive to head movement and camera quality
* Not medically certified
* Requires stable frame rate for reliable readings

---

## Security Considerations

* Passwords are hashed using Werkzeug
* Session-based authentication implemented
* No HTTPS or production-level security configuration

---

## Suggested Improvements

* Improve model accuracy using deep learning-based face tracking
* Add real-time graph visualization on frontend
* Implement REST API for scalability
* Use PostgreSQL instead of SQLite for production
* Deploy using Docker and cloud platforms
* Add multi-user analytics dashboard

---

## Purpose

This project demonstrates integration of:

* Computer Vision
* Signal Processing
* Web Development
* Database Management

Suitable for academic projects and intermediate-level portfolio demonstrations.

---

## License

This project is intended for educational and research purposes only.

## Output
<img width="1869" height="796" alt="image" src="https://github.com/user-attachments/assets/0c7ea36f-0a97-4304-8440-4eca060033c1" />
