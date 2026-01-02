import tensorflow as tf
import numpy as np
from tensorflow.keras import models,layers
# ==============================
# CONFIG
# ==============================
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 25
DATA_DIR = "potato_data"

# ==============================
# LOAD DATASET
# ==============================
dataset = tf.keras.preprocessing.image_dataset_from_directory(
    directory=DATA_DIR,
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=123
)

# SINGLE SOURCE OF TRUTH
class_names=dataset.class_names
print("CLASS ORDER:", class_names)


def get_dataset_partitions_tf(ds, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True, shuffle_size=10000):
    assert (train_split + test_split + val_split) == 1

    ds_size = len(ds)

    if shuffle:
        ds = ds.shuffle(shuffle_size, seed=12)

    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)

    train_ds = ds.take(train_size)
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)

    return train_ds, val_ds, test_ds

train_ds,val_ds,test_ds = get_dataset_partitions_tf(dataset)



train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

# ==============================
# PREPROCESSING
# ==============================
resize_and_rescale = tf.keras.Sequential([
    layers.Resizing(IMAGE_SIZE, IMAGE_SIZE),
    layers.Rescaling(1./255)
])


data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.05),
])

n_classes = 3
# ==============================
# MODEL
# ==============================
model = tf.keras.Sequential([
    # ONLY Input
    tf.keras.layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),

    # preprocessing layers
    resize_and_rescale,
    data_augmentation,

    # CNN
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(n_classes, activation='softmax')
])
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

history=model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20
)


# ==============================
# SAVE
# ==============================
model.save("my_model.keras")

with open("class_names.txt", "w") as f:
    for c in class_names:
        f.write(c + "\n")

print("MODEL SAVED WITH IMAGE_SIZE = 224")


