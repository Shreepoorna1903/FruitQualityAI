
# FruitQualityAI
Deep learning–based fresh vs. rotten fruit detection with a 9-layer Sequential CNN and transfer learning baselines (InceptionV3, ResNet50, VGG-16), plus a conveyor-belt + Arduino servo prototype for automated sorting.

# Fruits Quality Detection (Final Year Project)

A complete, end-to-end project that detects **fresh vs. rotten** fruits using Convolutional Neural Networks (CNNs) and transfer learning (ResNet50, VGG16, InceptionV3). It also includes a hardware prototype with a conveyor belt, camera capture, Arduino (servo) actuation, and serial control.

- Custom 9‑layer Sequential CNN for 10 classes (5 fruits × {fresh, rotten}).
- Transfer learning baselines: **ResNet50**, **VGG‑16**, **InceptionV3**.
- Training for up to **50 epochs** with checkpoints and learning‑rate scheduling.
- Jupyter/Colab notebooks + Python scripts for training and inference.
- Hardware pipeline: OpenCV capture → model inference → PySerial to Arduino → servo barrier on a belt.

> This repository is based on the project report *“Fruits Quality Detection”* (PES University, Aug–Dec 2022).

## Repository Structure

```
Fruits-Quality-Detection/
├─ README.md
├─ LICENSE
├─ requirements.txt
├─ .gitignore
├─ notebooks/
│  └─ 0_quickstart.ipynb
├─ src/
│  ├─ train_sequential.py
│  ├─ train_transfer.py
│  ├─ infer.py
│  └─ utils.py
├─ data/
│  ├─ README.md
│  └─ sample_images/
│     ├─ fresh/
│     └─ rotten/
├─ models/           # saved model .h5/.keras files, checkpoints, logs
└─ hardware/
   ├─ arduino_servo.ino
   └─ serial_control.py
```

## Quickstart

1. **Create and activate a Python environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Prepare dataset** (two-level folder structure):
   ```text
   data/
     train/
       apple_fresh/
       apple_rotten/
       banana_fresh/
       banana_rotten/
       mango_fresh/
       mango_rotten/
       papaya_fresh/
       papaya_rotten/
       pomegranate_fresh/
       pomegranate_rotten/
     val/            # optional (else a split is created from train)
     test/
   ```

3. **Train the sequential model**
   ```bash
   python src/train_sequential.py --train_dir data/train --val_dir data/val --epochs 50
   ```

4. **Train a transfer model (InceptionV3 / ResNet50 / VGG16)**
   ```bash
   python src/train_transfer.py --arch inception_v3 --train_dir data/train --val_dir data/val --epochs 50
   ```

5. **Run inference on an image or folder**
   ```bash
   python src/infer.py --model_path models/best_model.h5 --image /path/to/image.jpg
   ```

## Hardware Prototype

- Camera capture with OpenCV.
- Python script sends `0` (fresh) or `1` (rotten) over **PySerial**.
- Arduino sketch rotates a **servo** (e.g., to 90° for fresh, 180° for rotten) and can toggle a motor/relay.

See `hardware/serial_control.py` and `hardware/arduino_servo.ino`.

## Citation

If you use this code in academic or industry work, please cite the underlying report:

> Shreepoorna D Purohit, R Ashwin, R V N D Parthasarathy, Sneha Bhat, *Fruits Quality Detection*, PES University, Capstone Project Phase‑2, Aug–Dec 2022.

## License

MIT — see [LICENSE](LICENSE).
>>>>>>> 979cc90 (Initial commit: fruits quality detection (final year project))
