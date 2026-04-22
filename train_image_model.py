import json
import os
import random

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "data", "medical_images")
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "medical_image_model.keras")
LABELS_PATH = os.path.join(MODELS_DIR, "medical_image_labels.json")
METADATA_PATH = os.path.join(MODELS_DIR, "medical_image_metadata.json")

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
SEED = 42
INITIAL_EPOCHS = 8
FINE_TUNE_EPOCHS = 4
LEARNING_RATE = 1e-3
FINE_TUNE_LEARNING_RATE = 1e-5
VALIDATION_SPLIT = 0.2
ALLOWED_EXTENSIONS = {".bmp", ".gif", ".jpeg", ".jpg", ".png"}


def find_image_files(dataset_dir):
    image_files = []
    class_counts = {}

    for root, _, files in os.walk(dataset_dir):
        valid_files = [
            os.path.join(root, name)
            for name in files
            if os.path.splitext(name)[1].lower() in ALLOWED_EXTENSIONS
        ]

        if valid_files:
            class_name = os.path.relpath(root, dataset_dir).split(os.sep)[0]
            class_counts[class_name] = class_counts.get(class_name, 0) + len(valid_files)
            image_files.extend(valid_files)

    return image_files, class_counts


def validate_dataset_dir(dataset_dir):
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(
            f"Dataset folder not found: {dataset_dir}\n"
            "Create class folders like data/medical_images/skin_melanoma and add images inside them."
        )

    image_files, class_counts = find_image_files(dataset_dir)
    class_folders = [
        name for name in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, name))
    ]

    if not image_files:
        class_hint = ", ".join(class_folders) if class_folders else "no class folders yet"
        raise ValueError(
            "No medical training images were found.\n"
            f"Checked folder: {dataset_dir}\n"
            f"Detected class folders: {class_hint}\n\n"
            "Expected structure:\n"
            "data/medical_images/\n"
            "  skin_melanoma/\n"
            "    img1.jpg\n"
            "    img2.jpg\n"
            "  pneumonia_xray/\n"
            "    scan1.png\n"
            "  normal_xray/\n"
            "    scan2.png\n\n"
            "Allowed image formats: .bmp, .gif, .jpeg, .jpg, .png"
        )

    if len(class_counts) < 2:
        raise ValueError(
            "At least 2 classes are needed for training.\n"
            f"Found image counts: {class_counts}"
        )

    return image_files, class_counts


def load_and_preprocess_image(path, label):
    image_bytes = tf.io.read_file(path)
    image_tensor = tf.image.decode_image(image_bytes, channels=3, expand_animations=False)
    image_tensor = tf.image.resize(image_tensor, IMAGE_SIZE)
    image_tensor = tf.cast(image_tensor, tf.float32)
    return image_tensor, label


def build_datasets(dataset_dir):
    image_files, class_counts = find_image_files(dataset_dir)
    class_names = sorted(class_counts.keys())
    class_to_index = {name: index for index, name in enumerate(class_names)}

    samples = []
    for image_path in image_files:
        rel_parent = os.path.relpath(os.path.dirname(image_path), dataset_dir)
        class_name = rel_parent.split(os.sep)[0]
        samples.append((image_path, class_to_index[class_name]))

    random.Random(SEED).shuffle(samples)

    val_size = max(1, int(len(samples) * VALIDATION_SPLIT))
    if len(samples) - val_size < 1:
        val_size = max(0, len(samples) - 1)

    val_samples = samples[:val_size]
    train_samples = samples[val_size:]

    if not train_samples or not val_samples:
        raise ValueError(
            "Not enough images to create both training and validation splits.\n"
            f"Found {len(samples)} images total. Add more images per class and try again."
        )

    def make_dataset(split_samples, training=False):
        paths = [path for path, _ in split_samples]
        labels = [label for _, label in split_samples]
        labels = tf.keras.utils.to_categorical(labels, num_classes=len(class_names))

        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        ds = ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
        if training:
            ds = ds.shuffle(len(split_samples), seed=SEED)
        ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        return ds

    train_ds = make_dataset(train_samples, training=True)
    val_ds = make_dataset(val_samples, training=False)

    return train_ds, val_ds, class_names


def build_model(num_classes):
    augmentation = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.08),
            layers.RandomZoom(0.1),
            layers.RandomContrast(0.1),
        ],
        name="augmentation",
    )

    inputs = layers.Input(shape=(*IMAGE_SIZE, 3))
    x = augmentation(inputs)
    x = preprocess_input(x)

    base_model = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_tensor=x,
    )
    base_model.trainable = False

    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="medical_image_classifier")
    return model, base_model


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    image_files, class_counts = validate_dataset_dir(DATASET_DIR)
    print(f"Found {len(image_files)} training images across {len(class_counts)} classes.")
    for class_name, count in sorted(class_counts.items()):
        print(f"- {class_name}: {count} images")

    train_ds, val_ds, class_names = build_datasets(DATASET_DIR)
    model, base_model = build_model(len(class_names))

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=2,
            min_lr=1e-6,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
        ),
    ]

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    history_initial = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=INITIAL_EPOCHS,
        callbacks=callbacks,
    )

    base_model.trainable = True
    for layer in base_model.layers[:-20]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=FINE_TUNE_LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    history_finetune = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=INITIAL_EPOCHS + FINE_TUNE_EPOCHS,
        initial_epoch=len(history_initial.history["loss"]),
        callbacks=callbacks,
    )

    best_model = tf.keras.models.load_model(MODEL_PATH)
    loss, accuracy = best_model.evaluate(val_ds, verbose=0)

    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "image_size": IMAGE_SIZE,
                "class_names": class_names,
                "validation_accuracy": float(accuracy),
                "validation_loss": float(loss),
                "dataset_dir": DATASET_DIR,
                "base_model": "EfficientNetB0",
            },
            f,
            indent=2,
        )

    print(f"Medical image model saved to {MODEL_PATH}")
    print(f"Labels saved to {LABELS_PATH}")
    print(f"Validation accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
