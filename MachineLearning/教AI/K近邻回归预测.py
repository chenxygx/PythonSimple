import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

# 读取数据
df = pd.read_csv('../Data/RGZNXLY_data.csv')
# 特征处理。类别型特征转换独热编码形式。对Color和Type做独热编码
# 颜色独热编码One-hot
df_colors = df['Color'].str.get_dummies().add_prefix('Color: ')
# 类型独热编码
df_type = df['Type'].apply(str).str.get_dummies().add_prefix('Type: ')
# 拼接独热编码数据列
df = pd.concat([df, df_colors, df_type], axis=1)
# 去除独热编码对应的原始列
df = df.drop(['Brand', 'Type', 'Color'], axis=1)

# 数据转换，corr函数计算特征之间的相关性
# matrix = df.corr()
# f, ax = plt.subplots(figsize=(8, 6))
# sns.heatmap(matrix, square=True)
# plt.title('Car Price Variables')

# 特征归一化，标注来自训练数据
X = df[['Construction Year', 'Days Until MOT', 'Odometer']]
y = df['Ask Price'].values.reshape(-1, 1)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2003)
# 特征归一化，原始特征转换成均值为0方差为1的高斯分布
x_normalizer = StandardScaler()
x_train = x_normalizer.fit_transform(x_train)
x_test = x_normalizer.transform(x_test)

y_normalizer = StandardScaler()
y_train = y_normalizer.fit_transform(y_train)
y_test = y_normalizer.transform(y_test)

# 训练KNN模型，并进行预测。
knn = KNeighborsRegressor(n_neighbors=2)
knn.fit(x_train, y_train.ravel())
y_pred = knn.predict(x_test)
y_pred_inv = y_normalizer.inverse_transform(y_pred)
y_test_inv = y_normalizer.inverse_transform(y_test)

plt.scatter(y_pred_inv, y_test_inv)
plt.xlabel('Prediction')
plt.ylabel('Real value')

diagonal = np.linspace(500, 1500, 100)
plt.plot(diagonal, diagonal, '-r')
plt.xlabel('Predicted ask price')
plt.ylabel('Ask price')
# plt.show()

# 打印预测值
print(y_pred_inv)
print(y_test_inv)
