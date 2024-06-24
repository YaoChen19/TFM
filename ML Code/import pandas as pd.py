import pandas as pd
import numpy as np
from ast import literal_eval
from sklearn.preprocessing import LabelEncoder, StandardScaler
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Bidirectional, LSTM, Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.regularizers import l2

# 定义数据清理函数
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

# 读取数据并清理
data = pd.read_csv('E:/TFM/gestures.csv', header=None)
cleaned_data = clean_data(data)

# 读取老师的数据并清理
teacher_data = pd.read_csv('E:/TFM/gesturesProfe.csv', header=None)
cleaned_teacher_data = clean_data(teacher_data)

# 将你的数据和老师的数据分开处理
train_labels = cleaned_data['label']
train_x_data = cleaned_data['x']
train_y_data = cleaned_data['y']
train_z_data = cleaned_data['z']

test_labels = cleaned_teacher_data['label']
test_x_data = cleaned_teacher_data['x']
test_y_data = cleaned_teacher_data['y']
test_z_data = cleaned_teacher_data['z']

# 合并 x, y, z 数据
def merge_data(x_data, y_data, z_data):
    X = []
    for x, y, z in zip(x_data, y_data, z_data):
        combined = np.array([x, y, z]).T
        X.append(combined.tolist())
    return X

train_X = merge_data(train_x_data, train_y_data, train_z_data)
test_X = merge_data(test_x_data, test_y_data, test_z_data)

# 填充序列到相同长度
max_length = max([len(seq) for seq in train_X + test_X])
train_X = pad_sequences(train_X, maxlen=max_length, padding='post', dtype='float32')
test_X = pad_sequences(test_X, maxlen=max_length, padding='post', dtype='float32')

# 数据标准化
scaler = StandardScaler()
train_X_scaled = scaler.fit_transform(train_X.reshape(-1, train_X.shape[-1])).reshape(train_X.shape)
test_X_scaled = scaler.transform(test_X.reshape(-1, test_X.shape[-1])).reshape(test_X.shape)

# 转换标签为 one-hot 编码
label_encoder = LabelEncoder()
train_y = label_encoder.fit_transform(train_labels)
train_y_one_hot = to_categorical(train_y)

test_y = label_encoder.transform(test_labels)
test_y_one_hot = to_categorical(test_y)

# 构建混合模型
model = Sequential()
model.add(Conv1D(64, kernel_size=3, activation='relu', input_shape=(train_X_scaled.shape[1], train_X_scaled.shape[2]), kernel_regularizer=l2(0.001)))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(128, kernel_size=3, activation='relu', kernel_regularizer=l2(0.001)))
model.add(MaxPooling1D(pool_size=2))
model.add(Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=l2(0.001))))
model.add(Dropout(0.5))
model.add(Bidirectional(LSTM(64, kernel_regularizer=l2(0.001))))
model.add(Dropout(0.5))
model.add(Dense(train_y_one_hot.shape[1], activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

# 使用早停法和学习率调度器
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

# 训练模型
model.fit(train_X_scaled, train_y_one_hot, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stopping, reduce_lr])

# 评估模型在训练集和验证集上的表现
train_results = model.evaluate(train_X_scaled, train_y_one_hot)
print(f"Train Accuracy: {train_results[1] * 100:.2f}%")

# 评估模型在独立测试集上的表现
test_results = model.evaluate(test_X_scaled, test_y_one_hot)
print(f"Test Accuracy: {test_results[1] * 100:.2f}%")

# 打印分类报告
from sklearn.metrics import classification_report

y_pred_train = model.predict(train_X_scaled)
y_pred_test = model.predict(test_X_scaled)

y_pred_train_classes = np.argmax(y_pred_train, axis=1)
y_pred_test_classes = np.argmax(y_pred_test, axis=1)

y_train_classes = np.argmax(train_y_one_hot, axis=1)
y_test_classes = np.argmax(test_y_one_hot, axis=1)

print("Classification Report (Train):")
print(classification_report(y_train_classes, y_pred_train_classes, target_names=label_encoder.classes_))

print("Classification Report (Test):")
print(classification_report(y_test_classes, y_pred_test_classes, target_names=label_encoder.classes_))
