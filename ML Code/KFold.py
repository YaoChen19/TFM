import pandas as pd
import numpy as np
from ast import literal_eval
from sklearn.preprocessing import LabelEncoder, StandardScaler
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout
from keras.optimizers import Adam
from sklearn.model_selection import KFold
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils import shuffle
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

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

# 数据增强函数
def augment_data(data, labels):
    augmented_data = []
    augmented_labels = []
    for d, label in zip(data, labels):
        augmented_data.append(d)
        augmented_labels.append(label)
        # 添加噪声
        noise = np.random.normal(0, 0.1, d.shape)
        augmented_data.append(d + noise)
        augmented_labels.append(label)
        # 翻转数据
        augmented_data.append(np.flip(d, axis=0))
        augmented_labels.append(label)
    return np.array(augmented_data), np.array(augmented_labels)

# 读取训练数据并清理
data = pd.read_csv('E:/TFM/Train.csv', header=None)
cleaned_data = clean_data(data)

# 分开处理训练数据
labels = cleaned_data['label']
x_data = cleaned_data['x']
y_data = cleaned_data['y']
z_data = cleaned_data['z']

# 合并 x, y, z 数据
def merge_data(x_data, y_data, z_data):
    X = []
    for x, y, z in zip(x_data, y_data, z_data):
        combined = np.array([x, y, z]).T
        X.append(combined.tolist())
    return X

X = merge_data(x_data, y_data, z_data)

# 填充序列到相同长度
max_length = max([len(seq) for seq in X])
X = pad_sequences(X, maxlen=max_length, padding='post', dtype='float32')

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)

# 转换标签为 one-hot 编码
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)
y_one_hot = to_categorical(y)

# 数据增强
X_scaled, y_one_hot = augment_data(X_scaled, y_one_hot)

# 打乱数据
X_scaled, y_one_hot = shuffle(X_scaled, y_one_hot)

# 读取独立测试数据并清理
test_data = pd.read_csv('E:/TFM/Test.csv', header=None)
cleaned_test_data = clean_data(test_data)

# 分开处理测试数据
test_labels = cleaned_test_data['label']
test_x_data = cleaned_test_data['x']
test_y_data = cleaned_test_data['y']
test_z_data = cleaned_test_data['z']

test_X = merge_data(test_x_data, test_y_data, test_z_data)
test_X = pad_sequences(test_X, maxlen=max_length, padding='post', dtype='float32')
test_X_scaled = scaler.transform(test_X.reshape(-1, test_X.shape[-1])).reshape(test_X.shape)
test_y = label_encoder.transform(test_labels)
test_y_one_hot = to_categorical(test_y)

# 定义一个函数来绘制损失曲线
def plot_history(history):
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# 定义 KFold
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
accuracies = []

# 进行交叉验证
for train_index, val_index in kfold.split(X_scaled):
    X_train, X_val = X_scaled[train_index], X_scaled[val_index]
    y_train, y_val = y_one_hot[train_index], y_one_hot[val_index]

    model = Sequential()
    model.add(Conv1D(64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(128, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(64))
    model.add(Dropout(0.5))
    model.add(Dense(y_one_hot.shape[1], activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

    history = model.fit(X_train, y_train, epochs=100, batch_size=128, validation_data=(X_val, y_val), callbacks=[early_stopping, reduce_lr])

    val_accuracy = model.evaluate(X_val, y_val)[1]
    accuracies.append(val_accuracy)

    plot_history(history)

    y_pred_val = model.predict(X_val)
    y_pred_val_classes = np.argmax(y_pred_val, axis=1)
    y_val_classes = np.argmax(y_val, axis=1)

    cm = confusion_matrix(y_val_classes, y_pred_val_classes)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix for Fold {len(accuracies)}')
    plt.show()

print(f"Cross-validation accuracies: {accuracies}")
print(f"Mean accuracy: {np.mean(accuracies)}")

# 在独立测试集上评估模型
test_results = model.evaluate(test_X_scaled, test_y_one_hot)
print(f"Test Accuracy: {test_results[1] * 100:.2f}%")

# 打印分类报告
y_pred_test = model.predict(test_X_scaled)
y_pred_test_classes = np.argmax(y_pred_test, axis=1)
y_test_classes = np.argmax(test_y_one_hot, axis=1)

print("Classification Report (Test):")
print(classification_report(y_test_classes, y_pred_test_classes, target_names=label_encoder.classes_))

# 绘制测试集的混淆矩阵
cm_test = confusion_matrix(y_test_classes, y_pred_test_classes)
plt.figure(figsize=(10, 7))
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix (Test Set)')
plt.show()
