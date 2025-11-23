dataset_dir = r"C:\Users\shari\Downloads\flowers-recognition\flowers"
dataset_dir = pathlib.Path(dataset_dir)

# Load training and validation datasets
ds_train = tf.keras.utils.image_dataset_from_directory(
    dataset_dir,
    validation_split=VAL_SPLIT,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

ds_val = tf.keras.utils.image_dataset_from_directory(
    dataset_dir,
    validation_split=VAL_SPLIT,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# Get class names
class_names = ds_train.class_names
num_classes = len(class_names)

print("Flower classes:", class_names)
print("Number of classes:", num_classes)


plt.figure(figsize=(12, 8))
for images, labels in ds_train.take(1):  # Take 1 batch
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
plt.tight_layout()
plt.show()


def preprocess_image(image, label):
    """Preprocess images for MobileNetV3"""
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

def augment_image(image, label):
    """Data augmentation"""
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, 0.2)
    image = tf.image.random_contrast(image, 0.8, 1.2)
    image = tf.image.random_saturation(image, 0.8, 1.2)
    image = tf.image.random_hue(image, 0.1)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label


def prepare_dataset(dataset, batch_size=32, shuffle=False, augment=False):
    dataset = dataset.map(preprocess_image, num_parallel_calls=AUTOTUNE)

    if shuffle:
        dataset = dataset.shuffle(1000)

    if augment:
        dataset = dataset.map(augment_image, num_parallel_calls=AUTOTUNE)

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset

# Apply preprocessing
train_ds = prepare_dataset(ds_train, batch_size=BATCH_SIZE, shuffle=True, augment=True)
val_ds = prepare_dataset(ds_val, batch_size=BATCH_SIZE, shuffle=False, augment=False)

print("Datasets prepared successfully!")
print(f"Training batches: {tf.data.experimental.cardinality(train_ds)}")
print(f"Validation batches: {tf.data.experimental.cardinality(val_ds)}")


def augment_image_enhanced(image, label):
    """
    Improved data augmentation strategy (TensorFlow built-in only)
    """
    # Random flips
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)

    # Random brightness/contrast/saturation/hue
    image = tf.image.random_brightness(image, 0.2)
    image = tf.image.random_contrast(image, 0.8, 1.2)
    image = tf.image.random_saturation(image, 0.8, 1.2)
    image = tf.image.random_hue(image, 0.1)

    # Random 90-degree rotation (avoids needing tensorflow-addons)
    k = tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32)
    image = tf.image.rot90(image, k)

    # Random zoom (simulate with central crop + resize)
    scales = tf.random.uniform(shape=[], minval=0.8, maxval=1.0)
    crop_size = tf.cast(scales * tf.cast(tf.shape(image)[:2], tf.float32), tf.int32)
    image = tf.image.random_crop(image, size=[crop_size[0], crop_size[1], 3])
    image = tf.image.resize(image, IMG_SIZE)

    # Clip values
    image = tf.clip_by_value(image, 0.0, 1.0)

    return image, label


def augment_image_enhanced(image, label):
    """
    Improved data augmentation (TensorFlow only, safe for tf.data)
    """
    # Random flips
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)

    # Random brightness/contrast/saturation/hue
    image = tf.image.random_brightness(image, 0.2)
    image = tf.image.random_contrast(image, 0.8, 1.2)
    image = tf.image.random_saturation(image, 0.8, 1.2)
    image = tf.image.random_hue(image, 0.1)

    # Random 90-degree rotation
    k = tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32)
    image = tf.image.rot90(image, k)

    # --- FIXED ZOOM (crop & resize safely) ---
    # Get shape dynamically
    shape = tf.shape(image)
    h, w = shape[0], shape[1]

    # Random scale between 80%–100%
    scale = tf.random.uniform([], 0.8, 1.0)
    new_h = tf.cast(scale * tf.cast(h, tf.float32), tf.int32)
    new_w = tf.cast(scale * tf.cast(w, tf.float32), tf.int32)

    # Ensure crop size fits inside the image
    new_h = tf.maximum(new_h, 1)
    new_w = tf.maximum(new_w, 1)

    # Random crop
    image = tf.image.random_crop(image, size=[new_h, new_w, 3])
    # Resize back to target size
    image = tf.image.resize(image, IMG_SIZE)

    # Clip values
    image = tf.clip_by_value(image, 0.0, 1.0)

    return image, label


import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory

# Constants
IMG_SIZE = 224
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

# Path to dataset (adjust accordingly)
DATA_DIR = "flowers"

ds_train = image_dataset_from_directory(
    dataset_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=None
)

# Validation set (unbatched)
ds_val = image_dataset_from_directory(
    dataset_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=None
)

class_names = ds_train.class_names
print("Classes:", class_names)
random_rotation = tf.keras.layers.RandomRotation(factor=0.07, fill_mode="reflect")

def augment_image_enhanced(image, label):
    image = tf.squeeze(image)

    # Random crop (90%) then resize back
    new_h, new_w = int(0.9 * IMG_SIZE), int(0.9 * IMG_SIZE)
    image = tf.image.random_crop(image, size=[new_h, new_w, 3])
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])

    # Random flips
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)

    # Color jitter
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, 0.8, 1.2)

    # Rotation
    image = random_rotation(tf.expand_dims(image, axis=0))
    image = tf.squeeze(image, axis=0)

    return image, label

def create_generators(ds_train, ds_val, batch_size=BATCH_SIZE):
    # Training pipeline
    train_ds = ds_train.map(preprocess_image, num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.map(augment_image_enhanced, num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.shuffle(1000).batch(batch_size).prefetch(AUTOTUNE)

    # Validation pipeline
    val_ds = ds_val.map(preprocess_image, num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.batch(batch_size).prefetch(AUTOTUNE)

    return train_ds, val_ds

# Usage
train_ds, val_ds = create_generators(ds_train, ds_val, batch_size=BATCH_SIZE)

print("✅ Enhanced datasets prepared!")
print(f"Training batches: {tf.data.experimental.cardinality(train_ds)}")
print(f"Validation batches: {tf.data.experimental.cardinality(val_ds)}")
