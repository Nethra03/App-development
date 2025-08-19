import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
import os

# ✅ Load dataset info
dataset_name = "food101"
(ds_train, ds_test), info = tfds.load(
    dataset_name,
    split=["train", "validation"],
    with_info=True,
    as_supervised=True,
)

# ✅ Define Indian food classes (subset from Food-101)
INDIAN_CLASSES = [
    "butter_naan", "chicken_curry", "chole_bhature", "dal_makhani",
    "dosa", "idli", "paani_puri", "samosa"
]

# ✅ Convert class names to integer indices
INDIAN_LABELS = [info.features["label"].str2int(cls) for cls in INDIAN_CLASSES]

# ✅ Filter only Indian classes using numeric labels
def filter_classes(img, label):
    return tf.reduce_any(tf.equal(label, INDIAN_LABELS))

ds_train = ds_train.filter(filter_classes)
ds_test = ds_test.filter(filter_classes)

# ✅ Preprocess images
IMG_SIZE = 128
BATCH_SIZE = 32

def preprocess(img, label):
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0
    return img, label

ds_train = ds_train.map(preprocess).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
ds_test = ds_test.map(preprocess).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# ✅ Build model
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(len(INDIAN_CLASSES), activation="softmax")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# ✅ Train
EPOCHS = 5
history = model.fit(ds_train, validation_data=ds_test, epochs=EPOCHS)

# ✅ Save model
os.makedirs("../backend", exist_ok=True)
model.save("../backend/indian_food_model.h5")

print("✅ Model training complete! Saved to backend/indian_food_model.h5")