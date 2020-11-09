import numpy as np
import time
import tkinter as tk

UNIT = 150  # 像素
MAZE_H = 4  # 网络高度
MAZE_W = 4  # 网络宽度


class ForzenLakeEnv(tk.Tk, object):
    """
    走迷宫机器人可视化
    """

    def __init__(self, map):
        super(ForzenLakeEnv, self).__init__()
        self.map = map
        self.title('ForZenLake')
        self._build_env()

    def _build_env(self):
        """
        私有方法，初始化画布
        :return:
        """
        self.canvas = tk.Canvas(self, bg='white',
                                height=MAZE_H * UNIT,
                                width=MAZE_W * UNIT)
        # create grids
        for c in range(0, MAZE_W * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, MAZE_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, MAZE_H * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, MAZE_W * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)

        for i, m in enumerate(self.map):
            for j, v in enumerate(m):
                v = v.decode("utf-8")
                if v == 'H':
                    pass
                if v == 'S':
                    self.canvas.create_oval(120, 120, i * UNIT + 80, j * UNIT + 80,
                                            fill='green')
                    pass
                if v == 'G':
                    pass

        self.canvas.pack()

    def update(self):
        pass

    def render(self):
        time.sleep(0.1)
        self.update()
