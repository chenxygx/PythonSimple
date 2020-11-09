import numpy as np
import scipy.special as special


class neuralNetWork:
    """
    Python制作神经网络
    """

    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_grate):
        """
        设定输入层节点，隐藏层节点、输出层节点的数量
        :param input_nodes 输入层节点
        :param hidden_nodes 隐藏层节点
        :param output_nodes 输出层节点
        :param learning_grate 学习率
        """
        self.input = input_nodes
        self.hidden = hidden_nodes
        self.output = output_nodes
        self.lr = learning_grate
        # 连接权重，隐藏层和输出层
        self.wih = np.random.normal(0.0, pow(self.hidden, -0.5), (self.hidden, self.input))
        self.who = np.random.normal(0.0, pow(self.output, -0.5), (self.output, self.hidden))
        # 定义sigmoid损失函数当作阈值控制输出
        self.activation_function = lambda x: special.expit(x)

    def train(self, input_list, target_list):
        """
        训练任务，学习给定训练集样本化后，优化权重
        :param input_list 输入集合
        :param target_list 标签集合
        :return:
        """
        # 转换成二维数组
        inputs = np.array(input_list, ndmin=2).T
        targets = np.array(target_list, ndmin=2).T
        # 计算隐藏层
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        # 计算输出层
        final_inputs = np.dot(self.who, hidden_outputs)
        final_output = self.activation_function(final_inputs)

        # 反向传播，将输出值与期望值比对，用差值更新网络权重
        # 计算误差
        output_errors = targets - final_output
        hidden_errors = np.dot(self.who.T, output_errors)
        self.who += self.lr * np.dot((output_errors * final_output * (1.0 - final_output)),
                                     np.transpose(hidden_outputs))
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
                                     np.transpose(inputs))

    def query(self, inputs_list):
        """
        给定输入，从输出节点给出答案
        :param inputs_list: 输入矩阵
        :return 经过全连接网络的输出值
        """
        inputs = np.array(inputs_list, ndmin=2).T
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_output = self.activation_function(hidden_inputs)
        final_inputs = np.dot(self.who, hidden_output)
        final_output = self.activation_function(final_inputs)
        return final_output
