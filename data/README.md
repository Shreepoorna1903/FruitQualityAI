# Data README

This project expects an **image folder** dataset organized by class folders.

Example (10 classes = 5 fruits × {fresh, rotten}):
```
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
  val/    # optional
  test/
```

> Put a few images into `data/sample_images/` to quickly test `infer.py`.