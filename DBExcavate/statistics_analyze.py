import pandas as pd

catering_sale = 'data/catering_sale.xls'
data = pd.read_excel(catering_sale, index_col=u'日期')
data = data[(data[u'销量'] > 400) & (data[u'销量'] < 5000)]
statistics = data.describe()
statistics.loc['range'] = statistics.loc['max'] - statistics.loc['min']  # 极差
statistics.loc['var'] = statistics.loc['std'] / statistics.loc['mean']  # 变异系数
statistics.loc['dis'] = statistics.loc['75%'] - statistics.loc['25%']  # 四位分数间隔
print(statistics)