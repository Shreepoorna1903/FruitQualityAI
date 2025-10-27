import argparse, os, glob
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image

def load_and_preprocess(img_path, target_size):
    img = image.load_img(img_path, target_size=target_size)
    arr = image.img_to_array(img) / 255.0
    return np.expand_dims(arr, axis=0)

def main(args):
    model = keras.models.load_model(args.model_path)
    classes = None
    if args.class_indices and os.path.exists(args.class_indices):
        import json
        with open(args.class_indices) as f:
            idx = json.load(f)
            # invert
            classes = {int(v):k for k,v in idx.items()}

    paths = []
    if os.path.isdir(args.image):
        paths = sorted(glob.glob(os.path.join(args.image, "*.*")))
    else:
        paths = [args.image]

    target = model.input_shape[1:3]

    for p in paths:
        x = load_and_preprocess(p, target)
        probs = model.predict(x, verbose=0)[0]
        pred_idx = int(np.argmax(probs))
        pred_name = classes.get(pred_idx, str(pred_idx)) if classes else str(pred_idx)
        print(f"{os.path.basename(p)} -> {pred_name} (p={probs[pred_idx]:.3f})")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--image", required=True, help="Image path or folder")
    ap.add_argument("--class_indices", default="", help="Optional: JSON dumped from a Keras DirectoryIterator.class_indices")
    args = ap.parse_args()
    main(args)