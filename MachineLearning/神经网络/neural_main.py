import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from 神经网络.neural_network import neuralNetWork

input_nodes = 784
hidden_nodes = 100
output_nodes = 10
learning_rate = 0.2
# 调用神经网络模型
neural = neuralNetWork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# 训练手写码
train = pd.read_csv('./data/mnist_train_100.csv')
for e in range(100):
    for i in range(len(train)):
        # 颜色默认是0-255，缩小到0.01-1.0
        # 原始输入除以255，得到0-1范围输入。
        # 因为需要加上偏移量0.01。所以乘以0.99，变成0.0-0.99范围。
        inputs = (train.iloc[i, 1:].values / 255.0 * 0.99) + 0.01
        targets = np.zeros(output_nodes) + 0.01
        targets[int(train.iloc[i, 0])] = 0.99
        neural.train(inputs, targets)

# 测试手写码
test = pd.read_csv('./data/mnist_test_10.csv')
scorecard = []
for i in range(len(test)):
    correct_label = test.iloc[i, 0]
    inputs = (np.asfarray(test.iloc[i, 1:].values.ravel()) / 255.0 * 0.99) + 0.01
    output = neural.query(inputs)
    label = np.argmax(output)
    if label == correct_label:
        scorecard.append(1)
    else:
        scorecard.append(0)

scorecard_array = np.asarray(scorecard)
print("performance = ", scorecard_array.sum() / scorecard_array.size)

# train_one = train.iloc[3, 1:].values.reshape((28, 28))
# plt.imshow(scaled_input, cmap='Greys', interpolation='None')
# plt.show()
