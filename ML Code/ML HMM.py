import pandas as pd
import numpy as np
from ast import literal_eval
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA

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
data = pd.read_csv('E:/TFM/gesturesNOXR.csv', header=None)
cleaned_data = clean_data(data)

# 读取老师的数据并清理
teacher_data = pd.read_csv('E:/TFM/gesturesProfeNOXR.csv', header=None)
cleaned_teacher_data = clean_data(teacher_data)

# 分离训练数据和测试数据
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
    max_length = max([len(x) for x in x_data] + [len(y) for y in y_data] + [len(z) for z in z_data])
    X = []
    for x, y, z in zip(x_data, y_data, z_data):
        combined = np.zeros((max_length, 3))
        combined[:len(x), 0] = x
        combined[:len(y), 1] = y
        combined[:len(z), 2] = z
        X.append(combined)
    return np.array(X), max_length

train_X, max_length_train = merge_data(train_x_data, train_y_data, train_z_data)
test_X, max_length_test = merge_data(test_x_data, test_y_data, test_z_data)

# 确保长度一致
max_length = max(max_length_train, max_length_test)
train_X = np.array([np.pad(x, ((0, max_length - x.shape[0]), (0, 0)), 'constant') for x in train_X])
test_X = np.array([np.pad(x, ((0, max_length - x.shape[0]), (0, 0)), 'constant') for x in test_X])

# 展平序列以便缩放
train_X_flat = train_X.reshape(train_X.shape[0], -1)
test_X_flat = test_X.reshape(test_X.shape[0], -1)

# 特征缩放
scaler = StandardScaler()
train_X_scaled = scaler.fit_transform(train_X_flat)
test_X_scaled = scaler.transform(test_X_flat)

# 主成分分析 (PCA) 进行降维
pca = PCA(n_components=0.95)  # 保留95%的方差
train_X_pca = pca.fit_transform(train_X_scaled)
test_X_pca = pca.transform(test_X_scaled)

# 标签编码
label_encoder = LabelEncoder()
train_y = label_encoder.fit_transform(train_labels)
test_y = label_encoder.transform(test_labels)

# 训练SVM模型
svm_model = SVC(kernel='rbf', C=1, gamma='auto')
svm_model.fit(train_X_pca, train_y)

# 预测和评估
train_predictions = svm_model.predict(train_X_pca)
train_accuracy = accuracy_score(train_y, train_predictions)
print(f"Train Accuracy: {train_accuracy * 100:.2f}%")

test_predictions = svm_model.predict(test_X_pca)
test_accuracy = accuracy_score(test_y, test_predictions)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

print("Classification Report:")
print(classification_report(test_y, test_predictions, target_names=label_encoder.classes_))
