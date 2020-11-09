import numpy as np
import tensorflow as tf
import pandas as pd
from dense import dense_to_one_hot

# 加载数据集
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv").values

# 把图片数据取出来，进行处理
x_train = train.iloc[:, 1:].values
x_train = x_train.astype(np.float)

# 给到的图片的灰度数值在0~255，这里将图片的信息控制在0~1之间
x_train = np.multiply(x_train, 1.0 / 255.0)

# 计算图片的长和高，下面会用到
image_size = x_train.shape[1]
image_width = image_height = np.ceil(np.sqrt(image_size)).astype(np.uint8)

print('数据样本大小: (%g, %g)' % x_train.shape)
print('图片的维度大小 => {0}'.format(image_size))
print('图片长 => {0}\n图片高 => {1}'.format(image_width, image_height))

# 把数据集的标签结果取出来
labels_flat = train.iloc[:, 0].values.ravel()
labels_count = np.unique(labels_flat).shape[0]

# 对Label进行One_Hot处理
labels = dense_to_one_hot(labels_flat, labels_count)
labels = labels.astype(np.uint8)

print('标签({0[0]},{0[1]})'.format(labels.shape))
print('图片标签举例：[{0}] => {1}'.format(25, labels[25]))

# 把训练数据分为训练图片集和验证图片集

VALIDATION_SIZE = 2000

train_images = x_train[VALIDATION_SIZE:]
train_labels = labels[VALIDATION_SIZE:]

validation_images = x_train[:VALIDATION_SIZE]
validation_labels = labels[:VALIDATION_SIZE]

# 设置批次大小，求得批次数量
batch_size = 100
n_batch = int(len(train_images) / batch_size)

# 定义两个placeholder，用来承载数据，因为每个图片，都是一个784维的数据，所以我们的x是784列；
#  因为要把图片识别为0-9的10个数字，也就是有10个标签，所以y是10列
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])


#  定义几个处理的函数
def weight_variable(shape):
    # 初始化权重，正态分布 标准方差为0.1
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    # 初始化偏置值，设为非零避免死神经元
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

    # 对TensorFlow的2D卷积进行封装


def conv2d(x, W):
    # 卷积不改变输入的shape
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    # 对TensorFlow的池化进行封装


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


# 把输入变换成一个4d的张量，第二三个对应的是图片的长和宽，第四个参数对应的颜色
x_image = tf.reshape(x, [-1, 28, 28, 1])

# 计算32个特征，每3*3patch,第一二个参数指的是patch的size，第三个参数是输入的channels，第四个参数是输出的channels
W_conv1 = weight_variable([3, 3, 1, 32])

# 偏差的shape应该和输出的shape一致，所以也是32
b_conv1 = bias_variable([32])

# 28*28的图片卷积时步长为1，随意卷积后大小不变，按2*2最大值池化，相当于从2*2块中提取一个最大值，
# 所以池化后大小为[28/2,28/2] = [14,14]，第二次池化后为[14/2,14/2] = [7,7]

# 对数据做卷积操作
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

# 对结果做池化，max_pool_2x2之后，图片变成14*14
h_pool1 = max_pool_2x2(h_conv1)

# 在以前的基础上，生成了64个特征
W_conv2 = weight_variable([6, 6, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

# max_pool_2x2之后，图片变成7*7
h_pool2 = max_pool_2x2(h_conv2)
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])

# 构造一个全连接的神经网络，1024个神经元
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# 做Dropout操作
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 把1024个神经元的输入变为一个10维输出
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# 创建损失函数,以交叉熵的平均值为衡量
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_conv))

# 用梯度下降法优化参数
train_step_1 = tf.train.AdadeltaOptimizer(learning_rate=0.1).minimize(loss)

# 计算准确度
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_conv, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 设置保存模型的文件名参数
global_step = tf.Variable(0, name='global_step', trainable=False)
saver = tf.train.Saver()

# 初始化变量
init = tf.global_variables_initializer()

with tf.Session() as sess:
    # 初始化
    sess.run(init)

    # 这是载入以前训练好的模型的语句，有需要才用，注意把文件名改成成绩比较好的周期
    saver.restore(sess, "./model/model.ckpt-19")

    # 迭代20周期
    for epoch in range(19, 30):

        for batch in range(n_batch):
            # 每次取出一个数据块进行训练
            batch_x = train_images[(batch) * batch_size:(batch + 1) * batch_size]
            batch_y = train_labels[(batch) * batch_size:(batch + 1) * batch_size]

            # 【重要】 这是最终运行整个训练模型的语句
            sess.run(train_step_1, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5})

        # 每个周期计算一次准确度
        accuracy_n = sess.run(accuracy, feed_dict={x: validation_images, y: validation_labels, keep_prob: 1.0})

        print("第" + str(epoch + 1) + "轮，准确度为：" + str(accuracy_n))

        # 保存训练出来的模型，这样不用每次都从头开始训练了
        global_step.assign(epoch).eval()
        saver.save(sess, "./model/model.ckpt", global_step=global_step)

# 生成CSV
# with tf.Session() as sess:
#     sess.run(init)
#
#     saver.restore(sess, "model.ckpt-19")
#
#     test_x = np.array(test, dtype=np.float32)
#
#     conv_y_preditct = y_conv.eval(feed_dict={x: test_x[1:100, :], keep_prob: 1.0})
#
#     conv_y_preditct_all = list()
#     for i in np.arange(100, 28001, 100):
#         conv_y_preditct = y_conv.eval(feed_dict={x: test_x[i - 100:i, :], keep_prob: 1.0})
#
#         test_pred = np.argmax(conv_y_preditct, axis=1)
#
#         conv_y_preditct_all = np.append(conv_y_preditct_all, test_pred)
#
#     submission = pd.DataFrame({"ImageId": range(1, 28001), "Label": np.int32(conv_y_preditct_all)})
#
#     submission.to_csv("submission.csv", index=False)
