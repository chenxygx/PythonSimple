class tagObject(object):
    """
    选择状态属性
    """

    def __init__(self):
        self.weight = 0
        self.price = 0
        self.status = 0  # 0：未选中、1：已选中、2：已经不可选


class tagKnapsackProblem(object):
    """
    背包问题数据结构
    """

    def __init__(self):
        self.objs = tagObject[]
        self.totalC = 0


def GreedyAlgo(self, problem, spFunc):
    idx = 0
    ntc = 0
