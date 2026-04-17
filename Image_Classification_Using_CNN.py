import os

os.environ['KAGGLE_USERNAME'] = "pranavreddyatakula"
os.environ['KAGGLE_KEY'] = "KGAT_8f9230bfeead7cb96c5272353d85ace5"


!kaggle datasets list
!unzip DL.zip
!unzip test1.zip
!unzip train.zip
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras import mixed_precision

mixed_precision.set_global_policy('mixed_float16')

BASE_DIR = "/content"
TRAIN_PATH = "/content/train"
TEST_PATH = "/content/test1"
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 64
EPOCHS = 5
RANDOM_STATE = 42
print("GPUs Available: ", tf.config.list_physical_devices('GPU'))

filenames = os.listdir(TRAIN_PATH)
categories = []
for name in filenames:
    category = name.split('.')[0]
    categories.append(category)

df = pd.DataFrame({
    'filename': filenames,
    'category': categories
})

train_df, validate_df = train_test_split(
    df,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=df['category']
)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    zoom_range=0.2
)

train_generator = train_datagen.flow_from_dataframe(
    train_df,
    TRAIN_PATH,
    x_col='filename',
    y_col='category',
    target_size=IMAGE_SIZE,
    class_mode='binary',
    batch_size=BATCH_SIZE
)

validation_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = validation_datagen.flow_from_dataframe(
    validate_df,
    TRAIN_PATH,
    x_col='filename',
    y_col='category',
    target_size=IMAGE_SIZE,
    class_mode='binary',
    batch_size=BATCH_SIZE
)

model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(128,128,3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation="relu"),
    MaxPooling2D(2,2),

    GlobalAveragePooling2D(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(1, activation="sigmoid", dtype="float32")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator
)

test_filenames = os.listdir(TEST_PATH)
test_data = []

for filename in test_filenames:
    img = load_img(os.path.join(TEST_PATH, filename), target_size=IMAGE_SIZE)
    img_array = img_to_array(img) / 255.0
    test_data.append(img_array)

test_data = np.array(test_data)
predictions = model.predict(test_data)

# Convert probabilities to labels
predicted_labels = ['dog' if p > 0.5 else 'cat' for p in predictions]

results_df = pd.DataFrame({
    'filename': test_filenames,
    'label': predicted_labels
})

print(results_df.head())
