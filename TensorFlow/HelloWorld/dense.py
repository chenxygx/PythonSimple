import numpy as np


def dense_to_one_hot(labels_dense, num_classes):
    """
    将结果转换为矩阵形式
    :param labels_dense: 结果集合
    :param num_classes: 种类
    :return: 结果矩阵形式
    """
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot
