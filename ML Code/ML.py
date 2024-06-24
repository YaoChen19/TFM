import pandas as pd
import numpy as np
from ast import literal_eval
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# Defining Data Cleaning Functions
def clean_data(data):
    cleaned_data = []
    for i, row in data.iterrows():
        try:
            x = literal_eval(row[1])
            y = literal_eval(row[2])
            z = literal_eval(row[3])
            cleaned_data.append([row[0], x, y, z])
        except (ValueError, SyntaxError) as e:
            print(f"Skipping line {i + 1}: {e}")
    return pd.DataFrame(cleaned_data, columns=['label', 'x', 'y', 'z'])

# Read the data and clean it up
data = pd.read_csv('E:/TFM/gestures.csv', header=None)
cleaned_data = clean_data(data)

labels = cleaned_data['label']
x_data = cleaned_data['x']
y_data = cleaned_data['y']
z_data = cleaned_data['z']

# Merge x, y, z data
X = []
for x, y, z in zip(x_data, y_data, z_data):
    combined = np.array([x, y, z]).T
    X.append(combined.tolist())

# Fill sequences to the same length
max_length = max([len(seq) for seq in X])
X = pad_sequences(X, maxlen=max_length, padding='post', dtype='float32')

# Convert tags to one-hot encoding
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)
y_one_hot = to_categorical(y)

# Segmented data sets
X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)

# build a model
model = Sequential()
model.add(tf.keras.layers.Input(shape=(X.shape[1], X.shape[2])))
model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(128))
model.add(Dropout(0.5))
model.add(Dense(y_one_hot.shape[1], activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

from keras.preprocessing.sequence import TimeseriesGenerator

# Addition of data enhancements
data_gen = TimeseriesGenerator(X_train, y_train, length=max_length, batch_size=32)
# training model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# save model
model.save('gesture_model.keras')

# evaluate model
results = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {results[1] * 100:.2f}%")

# load keras model
model = tf.keras.models.load_model('E:/TFM/ML Code/gesture_model.keras')

# Converting models to TensorFlow Lite models
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
converter._experimental_lower_tensor_list_ops = False
tflite_model = converter.convert()

# Saving the converted model
try:
    tflite_model = converter.convert()
    with open('E:/TFM/ML Code/gesture_model.tflite', 'wb') as f:
       f.write(tflite_model)
    print("Model Conversion Successful")
except Exception as e:
    print(f"Model conversion failure: {e}")
