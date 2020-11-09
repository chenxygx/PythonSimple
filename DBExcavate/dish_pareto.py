import pandas as pd

dish_profit = 'data/catering_dish_profit.xls'
data = pd.read_excel(dish_profit, index_col=u'菜品名')
data = data[u'盈利'].copy()
data.sort_index(axis=0, ascending=False)

import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.figure()
plt.figure()
data.plot(kind='bar')
plt.ylabel(u'盈利（元）')
p = 1.0 * data.cumsum() / data.sum()
p.plot(color='r', secondary_y=True, style='-o', linewidth=2)
plt.annotate(format(p[6], '.4%'), xy=(6, p[6]), xytext=(6 * 0.9, p[6] * 0.9),
             arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
# 添加注释，即85%处的标记。这里包括了指定箭头样式。
plt.ylabel(u'盈利（比例）')
plt.show()
