import numpy as np
import pandas as pd
import tensorflow as tf
from dense import dense_to_one_hot

# 加载数据集，输入和结果拆分
train = pd.read_csv("./data/train.csv")
images = train.iloc[:, 1:].values
labels_flat = train.iloc[:, 0].values.ravel()

# 输入进行处理
images = images.astype(np.float)
images = np.multiply(images, 1.0 / 255.0)
print('输入数据的数量：(%g,%g)' % images.shape)

images_size = images.shape[1]
print('输入数据的维度：%s' % images_size)

# 结果进行处理
labels_count = np.unique(labels_flat).shape[0]
print('结果种类：%s' % labels_count)

# 输入数据划分训练集和验证集
labels = dense_to_one_hot(labels_flat, labels_count)
labels = labels.astype(np.uint8)

print('结果的数量：({0[0]},{0[1]})'.format(labels.shape))

VALIDATION_SIZE = 2000

validation_images = images[:VALIDATION_SIZE]
validation_labels = labels[:VALIDATION_SIZE]

train_images = images[VALIDATION_SIZE:]
train_labels = labels[VALIDATION_SIZE:]

# 训练集进行分配
batch_size = 100
n_batch = int(len(train_images) / batch_size)

# 创建神经网络对图片进行识别
x = tf.placeholder('float', shape=[None, images_size])
y = tf.placeholder("float", shape=[None, labels_count])

weights = tf.Variable(tf.zeros([784, 10]))
biases = tf.Variable(tf.zeros([10]))
result = tf.matmul(x, weights) + biases
prediction = tf.nn.softmax(result)

# 损失函数，计算交叉熵
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))

# 梯度下降算法
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# 初始化变量
init = tf.global_variables_initializer()

# 计算准确度
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    # 初始化
    sess.run(init)
    # 循环50轮
    for epoch in range(50):
        for batch in range(n_batch):
            # 按照分片取出数据
            batch_x = train_images[batch * batch_size:(batch + 1) * batch_size]
            batch_y = train_labels[batch * batch_size:(batch + 1) * batch_size]
            # 进行训练
            sess.run(train_step, feed_dict={x: batch_x, y: batch_y})
        # 每一轮计算一次准确度
        accuracy_n = sess.run(accuracy, feed_dict={x: validation_images, y: validation_labels})
        print("第" + str(epoch + 1) + "轮，准确度为：" + str(accuracy_n))
