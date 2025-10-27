import os, argparse
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping

def build_model(input_shape=(150,150,3), num_classes=10):
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D(),
        layers.Conv2D(128, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=keras.optimizers.Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def main(args):
    img_size = (150, 150)
    batch_size = args.batch_size

    train_datagen = keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        validation_split=0.0 if args.val_dir else 0.2,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest"
    )

    if args.val_dir:
        train_gen = train_datagen.flow_from_directory(
            args.train_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical'
        )
        val_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
        val_gen = val_datagen.flow_from_directory(
            args.val_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical'
        )
    else:
        train_gen = train_datagen.flow_from_directory(
            args.train_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical', subset='training'
        )
        val_gen = train_datagen.flow_from_directory(
            args.train_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical', subset='validation'
        )

    num_classes = train_gen.num_classes
    model = build_model(input_shape=(img_size[0], img_size[1], 3), num_classes=num_classes)

    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_path = os.path.join(args.out_dir, "best_sequential.h5")
    callbacks = [
        ReduceLROnPlateau(monitor="val_accuracy", factor=0.2, patience=7, verbose=1),
        ModelCheckpoint(ckpt_path, monitor="val_accuracy", save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_accuracy", patience=12, restore_best_weights=True, verbose=1)
    ]

    steps_per_epoch = None if args.steps_per_epoch <= 0 else args.steps_per_epoch
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        callbacks=callbacks,
        steps_per_epoch=steps_per_epoch
    )

    final_path = os.path.join(args.out_dir, "final_sequential.h5")
    model.save(final_path)
    print("Saved:", ckpt_path, "and", final_path)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train_dir", required=True)
    p.add_argument("--val_dir", default="")
    p.add_argument("--out_dir", default="models")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--steps_per_epoch", type=int, default=0, help="0 => auto")
    args = p.parse_args()
    main(args)