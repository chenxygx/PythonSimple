import pandas as pd
import numpy as np
import fifa_tools as fi
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# 加载数据
world_cup = pd.read_csv('datasets/World Cup 2018 Dataset.csv')
results = pd.read_csv('datasets/results.csv')

# 目标差异和结果列添加到结果数据集
winner = []
for i in range(len(results['home_team'])):
    if results['home_score'][i] > results['away_score'][i]:
        winner.append(results['home_team'][i])
    elif results['home_score'][i] < results['away_score'][i]:
        winner.append(results['away_team'][i])
    else:
        winner.append('Draw')
results['winner_team'] = winner

# 添加净胜球
results['goal_difference'] = np.absolute(results['home_score'] - results['away_score'])

results.to_csv('datasets/results2.csv')

# 优先提取一个国家进行分析，这里选择尼尔利亚，可以帮助我们找到那些特征对国家有效
df = results[(results['home_team'] == 'Nigeria') | (results['away_team'] == 'Nigeria')]
nigeria = df.iloc[:]

# 为年份创建一列，并选择所有1930年之后举行的比赛
year = []
for row in nigeria['date']:
    year.append(int(row[:4]))
nigeria['match_year'] = year
nigeria_1930 = nigeria[nigeria.match_year >= 1930]

# 创建所有参赛队伍
worldcup_teams = ['Russia', 'Saudi Arabia', 'Egypt', 'Uruguay', 'Porugal', 'Spain', 'Morocco', 'IRAN', 'France',
                  'Australia', 'Peru', 'Denmark', 'Argentina', 'Iceland', 'Croatia', 'Nigeria', 'Brazil', 'Switzerland',
                  'Costarica', 'Serbia', 'Germany', 'Mexico', 'Sweden', 'Korea', 'Belgium', 'Panama', 'Tunisia',
                  'England', 'Poland', 'Senegal', 'Columbia', 'Japan']

# 筛选从1930年起参加世界杯的队伍，并去掉重复的队伍
df_teams_home = results[results['home_team'].isin(worldcup_teams)]
df_teams_away = results[results['away_team'].isin(worldcup_teams)]
df_teams = pd.concat((df_teams_home, df_teams_away))
df_teams.drop_duplicates()

# 为年份创建一列，去掉1930年之前的比赛，并去掉不会影响比赛结果的数据列
year = []
for row in df_teams['date']:
    year.append(int(row[:4]))
df_teams['match_year'] = year
df_teams_1930 = df_teams[df_teams.match_year >= 1930]
df_teams_1930 = df_teams.drop(
    ['date', 'home_score', 'away_score', 'tournament', 'city', 'country', 'goal_difference', 'match_year'], axis=1)

# 为了简化模型，修改一下预测标签，如果主场获胜winning_team显示2，平局显示1，客场获胜显示0
df_teams_1930 = df_teams_1930.reset_index(drop=True)
df_teams_1930.loc[df_teams_1930.winner_team == df_teams_1930.home_team, 'winner_team'] = 2
df_teams_1930.loc[df_teams_1930.winner_team == 'Draw', 'winner_team'] = 1
df_teams_1930.loc[df_teams_1930.winner_team == df_teams_1930.away_team, 'winner_team'] = 0

# 将主场队伍和客场队伍从分类变量转换成连续的输入
final = pd.get_dummies(df_teams_1930, prefix=['home_team', 'away_team'], columns=['home_team', 'away_team'])

# 设置单独的x组和y组
X = final.drop(['winner_team'], axis=1)
y = final['winner_team']
y = y.astype('int')

# 采用训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# 采用逻辑回归算法
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
score = logreg.score(X_train, y_train)
score2 = logreg.score(X_test, y_test)

print('训练集的准确性：', '%.3f' % (score))
print('测试集的准确性：', '%.3f' % (score2))

# 加载新的数据集
ranking = pd.read_csv('datasets/fifa_rankings.csv')
fixtures = pd.read_csv('datasets/fixtures.csv')

pred_set = []
# 创建新的列，每个团队的排名
fixtures.insert(1, 'first_position', fixtures['Home Team'].map(ranking.set_index('Team')['Position']))
fixtures.insert(2, 'second_position', fixtures['Away Team'].map(ranking.set_index('Team')['Position']))
# 我们只需要小组阶段的排名，所以需要进行切片
fixtures = fixtures.iloc[:48, :]

# 根据每个团队的排名位置，将团队添加到新的预测数据集
for index, row in fixtures.iterrows():
    if row['first_position'] < row['second_position']:
        pred_set.append({'home_team': row['Home Team'], 'away_team': row['Away Team'], 'winner_team': None})
    else:
        pred_set.append({'home_team': row['Away Team'], 'away_team': row['Home Team'], 'winner_team': None})
pred_set = pd.DataFrame(pred_set)
backup_pred_set = pred_set
pred_set.head()

# 获取虚拟变量并删除winningteam列
pred_set = pd.get_dummies(pred_set, prefix=['home_team', 'away_team'], columns=['home_team', 'away_team'])

# 与模型的训练数据集相比，添加缺失的列
missing_cols = set(final.columns) - set(pred_set.columns)
for c in missing_cols:
    pred_set[c] = 0
pred_set = pred_set[final.columns]

# 删除获胜团队列
pred_set = pred_set.drop(['winner_team'], axis=1)

# 小组赛
print('---开始进行小组赛预测---')
predictions = logreg.predict(pred_set)
for i in range(fixtures.shape[0]):
    print(fi.team_name[backup_pred_set.iloc[i, 1]] + " vs " + fi.team_name[backup_pred_set.iloc[i, 0]])
    if predictions[i] == 2:
        print("胜利：" + fi.team_name[backup_pred_set.iloc[i, 1]])
    elif predictions[i] == 1:
        print("平局")
    elif predictions[i] == 0:
        print("胜利：" + fi.team_name[backup_pred_set.iloc[i, 0]])
    print(fi.team_name[backup_pred_set.iloc[i, 1]] + ' 胜利概率: ', '%.3f' % (logreg.predict_proba(pred_set)[i][2]))
    print('平局概率 %.3f' % (logreg.predict_proba(pred_set)[i][1]))
    print(fi.team_name[backup_pred_set.iloc[i, 0]] + ' 胜利概率: ', '%.3f' % (logreg.predict_proba(pred_set)[i][0]))
    print("")

# 16强
# print('---开始进行八分之一决赛预测---')
# group_16 = [('Uruguay', 'Portugal'), ('France', 'Croatia'), ('Brazil', 'Mexico'), ('England', 'Colombia'),
#             ('Spain', 'Russia'), ('Argentina', 'Peru'), ('Germany', 'Switzerland'), ('Poland', 'Belgium')]
# fi.clean_and_predict(group_16, ranking, final, logreg)

# 8强
# print('---开始进行半准决赛预测---')
# group_8 = [('Portugal', 'France'), ('Spain', 'Argentina'), ('Brazil', 'England'), ('Germany', 'Belgium')]
# fi.clean_and_predict(group_8, ranking, final, logreg)

# 4分之一决赛
# print('---开始进行半决赛预测---')
# group_4 = [('Portugal', 'Brazil'), ('Argentina', 'Germany')]
# fi.clean_and_predict(group_4, ranking, final, logreg)

# 决赛
# print('---开始进行决赛预测---')
# group_2 = [('Brazil', 'Germany')]
# fi.clean_and_predict(group_2, ranking, final, logreg)