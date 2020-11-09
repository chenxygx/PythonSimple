class Node():
    def __init__(self, data):
        self.data = data
        self.next = None


# 头插法
class SingnalNode():
    def __init__(self):
        self.current_node = None

    def add_node(self, data):
        """
        头插法插入节点
        :param data:
        :return:
        """
        node = Node(data)
        node.next = self.current_node
        self.current_node = node

    def append_node(self, data):
        """
        尾插法插入节点
        :param data:
        :return:
        """
        node = Node(data)
        cur = self.current_node
        # 遍历链表直到头节点处停止遍历
        while cur:
            if cur.next == None:
                break
            cur = cur.next
        cur.next = node

    def travel(self):
        """
        遍历链表
        :return:
        """
        cur = self.current_node
        while cur:
            print(cur.data)
            cur = cur.next

    def is_empty(self):
        """
        判断链表非空
        :return:
        """
        return self.current_node == None

    def get_lenth(self):
        """
        获取链表的长度
        :return:
        """
        cur = self.current_node
        count = 0
        while cur:
            count += 1
            cur = cur.next
        return count

    def insert_node(self, index, data):
        """
        指定位置插入节点
        :param index:
        :param data:
        :return:
        """
        link_len = self.get_lenth()
        if index == 0:
            self.add_node(data)
        elif index >= link_len:
            self.append_node(data)
        else:
            cur = self.current_node
            for i in range(1, index):
                cur = cur.next
            node = Node(data)
            node.next = cur.next
            cur.next = node

    def del_node(self, index):
        """
        根据索引删除节点
        :param index:
        :return:
        """
        # 找到前节点
        cur = self.current_node
        # 前驱节点
        pre = None
        count = 1
        len_num = self.get_lenth()
        while cur:
            if index == 1:
                self.current_node = cur.next
                break
            if count == index and count < len_num:
                pre.next = cur.next
                break
            if count >= len_num:
                pre.next = None
                break
            count += 1
            pre = cur
            cur = cur.next

    def del_node(self, index):
        """
        根据索引删除节点
        :param index:
        :return:
        """
        # 找到前节点
        cur = self.current_node
        if index == 1:
            self.current_node = cur.next
            return
        # 前驱节点
        pre = None
        len_num = self.get_lenth()
        for i in range(1, len_num):
            if i == index:
                break
            pre = cur
            cur = cur.next
        pre.next = cur.next


if __name__ == "__main__":
    test = SingnalNode()
    list_data = [1, 2, 3]
    for i in list_data:
        test.add_node(i)
    test.travel()
    # print(test.is_empty())
    # print (test.get_lenth())
    # test.append_node(4)
    # test.insert_node(1, 4)
    # test.travel()
    test.del_node(4)
    test.travel()
