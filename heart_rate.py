import cv2
import numpy as np
import time
import collections

# =========================
# GLOBAL STATE (single-user version)
# =========================
buffer_size = 150
buffer_r = []
buffer_g = []
times = []
bpm = 0
bpm_history = collections.deque(maxlen=10)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# =========================
# SIGNAL PROCESSING
# =========================
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = max(0.01, min(0.95, lowcut / nyq))
    high = max(low + 0.01, min(0.99, highcut / nyq))
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs):
    b, a = butter_bandpass(lowcut, highcut, fs)
    return filtfilt(b, a, data)


# =========================
# 🔥 MAIN FUNCTION (IMPORTANT)
# =========================
def process_frame(frame):
    global buffer_g, times

    import time
    import numpy as np
    import cv2

    frame = cv2.resize(frame, (320, 240))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.2, 5)

    if len(faces) == 0:
        return 0

    (x, y, w, h) = faces[0]

    roi = frame[y:y+h//2, x:x+w]

    g = np.mean(roi[:, :, 1])

    buffer_g.append(g)
    times.append(time.time())

    # keep buffer
    if len(buffer_g) > 100:
        buffer_g = buffer_g[-100:]
        times = times[-100:]

    # need enough data
    if len(buffer_g) < 50:
        return 0

    signal = np.array(buffer_g)
    signal = signal - np.mean(signal)

    # simple peak count
    peaks = np.where(signal > np.std(signal))[0]

    duration = times[-1] - times[0]

    if duration == 0:
        return 0

    bpm = (len(peaks) / duration) * 60

    if 40 <= bpm <= 180:
        return int(bpm)

    return 0

    try:
        duration = times[-1] - times[0]
        fs = len(times) / duration if duration > 0 else 30

        signal = np.array(buffer_g) - np.mean(buffer_g)

        # Smooth
        if len(signal) > 7:
            signal = savgol_filter(signal, 7, 2)

        # Filter
        filtered = butter_bandpass_filter(signal, 0.67, 4.0, fs)

        # FFT
        fft = np.abs(np.fft.rfft(filtered))
        freqs = np.fft.rfftfreq(len(filtered), d=1/fs)

        mask = (freqs >= 0.67) & (freqs <= 4.0)

        if np.any(mask):
            peak_freq = freqs[mask][np.argmax(fft[mask])]
            bpm_val = peak_freq * 60
        else:
            bpm_val = 0

        # Validate BPM
        if 40 <= bpm_val <= 200:
            bpm_history.append(bpm_val)

            weights = np.linspace(0.5, 1.0, len(bpm_history))
            bpm = int(np.average(bpm_history, weights=weights))

        return bpm

    except Exception as e:
        print(f"[ERROR] Signal processing failed: {e}")
        return 0
