import pickle
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

# ── Load Keras model & scaler ────────────────────────────────────────────────
model = tf.keras.models.load_model('./model.keras')

scaler_dict  = pickle.load(open('./scaler.pkl', 'rb'))
scaler       = scaler_dict['scaler']
label_encoder = scaler_dict['label_encoder']

# ── Camera setup ─────────────────────────────────────────────────────────────
import os

def open_camera():
    candidates = [
        (1, cv2.CAP_DSHOW),
        (2, cv2.CAP_DSHOW),
        (1, cv2.CAP_ANY),
        (2, cv2.CAP_ANY),
        (0, cv2.CAP_DSHOW),
        (0, cv2.CAP_ANY),
    ]
    for index, backend in candidates:
        cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            cap.release()
            continue
        ret, frame = cap.read()
        if ret and frame is not None:
            print(f"Using camera index={index}, backend={backend}")
            return cap
        cap.release()
    return None

cap = open_camera()
if cap is None:
    raise Exception("Error: Could not find any working camera.")

# ── MediaPipe ─────────────────────────────────────────────────────────────────
mp_hands         = mp.solutions.hands
mp_drawing       = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# ── Confidence threshold ──────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.60   # Show letter only if model is ≥60% sure

# ── Main loop ─────────────────────────────────────────────────────────────────
while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    if not ret:
        continue

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results   = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        # Draw landmarks for all detected hands
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        # Only classify the first hand
        hand_landmarks = results.multi_hand_landmarks[0]

        for i in range(len(hand_landmarks.landmark)):
            x_.append(hand_landmarks.landmark[i].x)
            y_.append(hand_landmarks.landmark[i].y)

        # Scale-invariant normalisation (bounding-box)
        min_x, min_y = min(x_), min(y_)
        max_span = max(max(x_) - min_x, max(y_) - min_y)
        max_span = max(max_span, 1e-6)

        for i in range(len(hand_landmarks.landmark)):
            data_aux.append((hand_landmarks.landmark[i].x - min_x) / max_span)
            data_aux.append((hand_landmarks.landmark[i].y - min_y) / max_span)

        # Bounding box coordinates for the display rectangle
        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10
        x2 = int(max(x_) * W) + 10
        y2 = int(max(y_) * H) + 10

        # ── Predict ──────────────────────────────────────────────────────────
        feature_vec  = scaler.transform([np.asarray(data_aux, dtype=np.float32)])
        probabilities = model.predict(feature_vec, verbose=0)[0]   # shape: (26,)

        max_confidence = float(np.max(probabilities))
        predicted_idx  = int(np.argmax(probabilities))

        # Convert index back to letter via LabelEncoder
        predicted_label = label_encoder.inverse_transform([predicted_idx])[0]
        predicted_char  = chr(65 + int(predicted_label))   # '0'→'A', '1'→'B', …

        # ── Display ───────────────────────────────────────────────────────────
        if max_confidence >= CONFIDENCE_THRESHOLD:
            display_text = f"{predicted_char}  {max_confidence*100:.0f}%"
            text_color   = (0, 0, 0)       # Black — confident
            box_color    = (0, 200, 0)     # Green box
        else:
            display_text = f"?  {max_confidence*100:.0f}%"
            text_color   = (0, 0, 220)     # Red — uncertain
            box_color    = (0, 0, 200)     # Red box

        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 3)
        cv2.putText(frame, display_text, (x1, y1 - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, text_color, 3, cv2.LINE_AA)

    cv2.imshow('ASL Detector', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
