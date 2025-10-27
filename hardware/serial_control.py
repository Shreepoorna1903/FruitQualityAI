import argparse, time, os
import serial
import cv2
import numpy as np
from tensorflow import keras

def send_command(port: str, value: int, baud: int=9600):
    with serial.Serial(port, baudrate=baud, timeout=2) as ser:
        ser.write(b"1" if value else b"0")
        ser.flush()
        time.sleep(0.2)

def capture_image(device=0, out_path="capture.jpg", width=640, height=480):
    cap = cv2.VideoCapture(device)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    time.sleep(0.5)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Failed to capture frame.")
    cv2.imwrite(out_path, frame)
    return out_path

def classify(model_path, img_path):
    model = keras.models.load_model(model_path)
    target = model.input_shape[1:3]
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target)
    x = img.astype(np.float32) / 255.0
    x = np.expand_dims(x, 0)
    probs = model.predict(x, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    return pred_idx, float(probs[pred_idx])

def main(args):
    # Optionally capture a photo
    img_path = args.image or capture_image(device=args.camera_index, out_path="capture.jpg")
    pred_idx, conf = classify(args.model_path, img_path)

    # Heuristic: assume class names encode 'fresh'/'rotten' in even/odd mapping if provided as 10 classes
    # For a real deployment, map indices using saved class_indices.
    is_rotten = (pred_idx % 2 == 1) if args.rotten_odd else (pred_idx % 2 == 0)

    print(f"Prediction idx={pred_idx}, confidence={conf:.3f} -> {'ROTTEN' if is_rotten else 'FRESH'}")
    send_command(args.serial_port, 1 if is_rotten else 0, baud=args.baud)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--serial_port", required=True, help="e.g., COM3 (Windows) or /dev/ttyUSB0 (Linux/Mac)")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--image", default="", help="Optional path to an existing image; else the camera is used.")
    ap.add_argument("--camera_index", type=int, default=0)
    ap.add_argument("--baud", type=int, default=9600)
    ap.add_argument("--rotten_odd", action="store_true", help="If class indices are arranged so that rotten classes are odd numbers")
    args = ap.parse_args()
    main(args)