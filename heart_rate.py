import cv2
import numpy as np
import time
import collections

# =========================
# GLOBAL STATE
# =========================
buffer_g = []
times = []

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)


def process_frame(frame):
    global buffer_g, times

    frame = cv2.resize(frame, (320, 240))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.2, 5)

    if len(faces) == 0:
        return 0

    (x, y, w, h) = faces[0]

    # Forehead ROI
    roi = frame[y:y + h//2, x:x + w]

    g = np.mean(roi[:, :, 1])

    buffer_g.append(g)
    times.append(time.time())

    # keep last 100 samples
    if len(buffer_g) > 100:
        buffer_g = buffer_g[-100:]
        times = times[-100:]

    # need enough data
    if len(buffer_g) < 50:
        return 0

    signal = np.array(buffer_g)
    signal = signal - np.mean(signal)

    # simple smoothing
    signal = np.convolve(signal, np.ones(5)/5, mode='valid')

    # peak detection (basic but better)
    peaks = []
    for i in range(1, len(signal)-1):
        if signal[i] > signal[i-1] and signal[i] > signal[i+1]:
            peaks.append(i)

    duration = times[-1] - times[0]

    if duration <= 0:
        return 0

    bpm = (len(peaks) / duration) * 60

    if 40 <= bpm <= 180:
        return int(bpm)

    return 0
