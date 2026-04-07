"""
sign_server.py — Sign Language WebSocket Server
================================================
Receives JPEG frames from the browser over WebSocket,
runs MediaPipe Hands + the Keras ASL model, and streams
{letter, confidence} JSON back to the client.

Run:
    python sign_server.py
Server listens on ws://localhost:8765
"""

import asyncio
import json
import os
import warnings

import cv2
import mediapipe as mp
import numpy as np
import pickle
import tensorflow as tf
import websockets
import websockets.exceptions

warnings.filterwarnings("ignore")

# ── Paths ───────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Load model & scaler ─────────────────────────────────────────────────────────
print("🤟  Sign-Language WebSocket Server")
print("   Loading Keras model …")
model = tf.keras.models.load_model(os.path.join(BASE_DIR, "model.keras"))

print("   Loading scaler …")
scaler_dict   = pickle.load(open(os.path.join(BASE_DIR, "scaler.pkl"), "rb"))
scaler        = scaler_dict["scaler"]
label_encoder = scaler_dict["label_encoder"]

# ── MediaPipe ───────────────────────────────────────────────────────────────────
print("   Initialising MediaPipe …")
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.50,
    min_tracking_confidence=0.50,
)

CONFIDENCE_THRESHOLD = 0.60


# ── Core inference (blocking — runs in executor thread) ─────────────────────────
def process_frame(frame_bytes: bytes) -> dict:
    """Decode JPEG bytes → MediaPipe → model → return {letter, confidence}."""
    nparr = np.frombuffer(frame_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return {"letter": None, "confidence": 0.0}

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results   = hands.process(frame_rgb)

    if not results.multi_hand_landmarks:
        return {"letter": None, "confidence": 0.0}

    lms   = results.multi_hand_landmarks[0].landmark
    x_    = [lm.x for lm in lms]
    y_    = [lm.y for lm in lms]

    min_x, min_y = min(x_), min(y_)
    max_span     = max(max(x_) - min_x, max(y_) - min_y, 1e-6)

    data_aux = []
    for lm in lms:
        data_aux.append((lm.x - min_x) / max_span)
        data_aux.append((lm.y - min_y) / max_span)

    feature_vec   = scaler.transform([np.asarray(data_aux, dtype=np.float32)])
    probabilities = model.predict(feature_vec, verbose=0)[0]

    max_conf  = float(np.max(probabilities))
    pred_idx  = int(np.argmax(probabilities))

    if max_conf >= CONFIDENCE_THRESHOLD:
        pred_label = label_encoder.inverse_transform([pred_idx])[0]
        pred_char  = chr(65 + int(pred_label))   # index → 'A', 'B', …
        return {"letter": pred_char, "confidence": round(max_conf, 3)}

    return {"letter": None, "confidence": round(max_conf, 3)}


# ── WebSocket handler ───────────────────────────────────────────────────────────
async def handler(websocket):
    addr = websocket.remote_address
    print(f"✅  Client connected:    {addr}")
    loop = asyncio.get_event_loop()
    busy = False   # per-connection frame-skip flag

    try:
        async for message in websocket:
            if not isinstance(message, bytes):
                continue          # ignore text frames
            if busy:
                continue          # drop frame while model is running

            busy = True
            try:
                result = await loop.run_in_executor(None, process_frame, message)
                await websocket.send(json.dumps(result))
            except Exception as exc:
                print(f"   Frame error: {exc}")
                try:
                    await websocket.send(json.dumps({"letter": None, "confidence": 0.0}))
                except Exception:
                    pass
            finally:
                busy = False

    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        print(f"❌  Client disconnected: {addr}")


# ── Entry point ─────────────────────────────────────────────────────────────────
async def main():
    print("✅  Server ready  →  ws://localhost:8765\n")
    async with websockets.serve(handler, "localhost", 8765, max_size=10_000_000):
        await asyncio.Future()   # run forever


if __name__ == "__main__":
    asyncio.run(main())
