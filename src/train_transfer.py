import os, argparse
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import InceptionV3, ResNet50, VGG16
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_pre
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_pre
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_pre
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping

def get_backbone(arch: str, input_shape):
    arch = arch.lower()
    if arch == "inception_v3":
        return InceptionV3(include_top=False, weights="imagenet", input_shape=input_shape), inception_pre
    if arch == "resnet50":
        return ResNet50(include_top=False, weights="imagenet", input_shape=input_shape), resnet_pre
    if arch == "vgg16":
        return VGG16(include_top=False, weights="imagenet", input_shape=input_shape), vgg_pre
    raise ValueError(f"Unknown arch: {arch}")

def build_transfer(arch="inception_v3", input_shape=(299,299,3), num_classes=10, train_backbone=False):
    backbone, _ = get_backbone(arch, input_shape)
    backbone.trainable = train_backbone

    x = layers.Input(shape=input_shape)
    y = backbone(x, training=False)
    y = layers.GlobalAveragePooling2D()(y)
    y = layers.Dropout(0.2)(y)
    y = layers.Dense(256, activation="relu")(y)
    out = layers.Dense(num_classes, activation="softmax")(y)
    model = keras.Model(inputs=x, outputs=out)
    model.compile(optimizer=keras.optimizers.Adam(),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model

def main(args):
    arch = args.arch.lower()
    img_size = (299, 299) if arch == "inception_v3" else (224, 224)
    input_shape = (*img_size, 3)

    # Choose preprocessing function
    _, pre = get_backbone(arch, input_shape)

    train_datagen = keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=pre,
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
            args.train_dir, target_size=img_size, batch_size=args.batch_size, class_mode='categorical'
        )
        val_datagen = keras.preprocessing.image.ImageDataGenerator(preprocessing_function=pre)
        val_gen = val_datagen.flow_from_directory(
            args.val_dir, target_size=img_size, batch_size=args.batch_size, class_mode='categorical'
        )
    else:
        train_gen = train_datagen.flow_from_directory(
            args.train_dir, target_size=img_size, batch_size=args.batch_size, class_mode='categorical', subset='training'
        )
        val_gen = train_datagen.flow_from_directory(
            args.train_dir, target_size=img_size, batch_size=args.batch_size, class_mode='categorical', subset='validation'
        )

    num_classes = train_gen.num_classes
    model = build_transfer(arch=arch, input_shape=input_shape, num_classes=num_classes, train_backbone=args.train_backbone)

    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_path = os.path.join(args.out_dir, f"best_{arch}.h5")
    callbacks = [
        ReduceLROnPlateau(monitor="val_accuracy", factor=0.2, patience=7, verbose=1),
        ModelCheckpoint(ckpt_path, monitor="val_accuracy", save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_accuracy", patience=12, restore_best_weights=True, verbose=1)
    ]

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        callbacks=callbacks
    )

    final_path = os.path.join(args.out_dir, f"final_{arch}.h5")
    model.save(final_path)
    print("Saved:", ckpt_path, "and", final_path)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--arch", default="inception_v3", choices=["inception_v3","resnet50","vgg16"])
    p.add_argument("--train_dir", required=True)
    p.add_argument("--val_dir", default="")
    p.add_argument("--out_dir", default="models")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--train_backbone", action="store_true")
    args = p.parse_args()
    main(args)