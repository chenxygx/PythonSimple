import pandas as pd


def clean_and_predict(matches, ranking, final, logreg):
    # Initialization of auxiliary list for data cleaning
    positions = []

    # Loop to retrieve each team's position according to FIFA ranking
    for match in matches:
        positions.append(ranking.loc[ranking['Team'] == match[0], 'Position'].iloc[0])
        positions.append(ranking.loc[ranking['Team'] == match[1], 'Position'].iloc[0])

    # Creating the DataFrame for prediction
    pred_set = []

    # Initializing iterators for while loop
    i = 0
    j = 0

    # 'i' will be the iterator for the 'positions' list, and 'j' for the list of matches (list of tuples)
    while i < len(positions):
        dict1 = {}

        # If position of first team is better, he will be the 'home' team, and vice-versa
        if positions[i] < positions[i + 1]:
            dict1.update({'home_team': matches[j][0], 'away_team': matches[j][1]})
        else:
            dict1.update({'home_team': matches[j][1], 'away_team': matches[j][0]})

        # Append updated dictionary to the list, that will later be converted into a DataFrame
        pred_set.append(dict1)
        i += 2
        j += 1

    # Convert list into DataFrame
    pred_set = pd.DataFrame(pred_set)
    backup_pred_set = pred_set

    # Get dummy variables and drop winning_team column
    pred_set = pd.get_dummies(pred_set, prefix=['home_team', 'away_team'], columns=['home_team', 'away_team'])

    # Add missing columns compared to the model's training dataset
    missing_cols2 = set(final.columns) - set(pred_set.columns)
    for c in missing_cols2:
        pred_set[c] = 0
    pred_set = pred_set[final.columns]

    # Remove winning team column
    pred_set = pred_set.drop(['winner_team'], axis=1)

    # Predict!
    predictions = logreg.predict(pred_set)
    for i in range(len(pred_set)):
        print(team_name[backup_pred_set.iloc[i, 1]] + " vs " + team_name[backup_pred_set.iloc[i, 0]])
        if predictions[i] == 2:
            print("胜利： " + team_name[backup_pred_set.iloc[i, 1]])
        elif predictions[i] == 1:
            print("平局")
        elif predictions[i] == 0:
            print("胜利： " + team_name[backup_pred_set.iloc[i, 0]])
        print(team_name[backup_pred_set.iloc[i, 1]] + ' 胜利概率： ',
              '%.3f' % (logreg.predict_proba(pred_set)[i][2]))
        print('平局概率 %.3f' % (logreg.predict_proba(pred_set)[i][1]))
        print(team_name[backup_pred_set.iloc[i, 0]] + ' 胜利概率： ',
              '%.3f' % (logreg.predict_proba(pred_set)[i][0]))
        print("")


team_name = {'Russia': '俄罗斯', 'Saudi Arabia': '沙特阿拉伯', 'Egypt': '埃及', 'Uruguay': '乌拉圭', 'Portugal': '葡萄牙',
    'Spain': '西班牙', 'Morocco': '摩洛哥', 'Iran': '伊朗', 'France': '法国', 'Australia': '澳大利亚', 'Peru': '秘鲁', 'Denmark': '丹麦',
    'Argentina': '阿根廷', 'Iceland': '冰岛', 'Croatia': '克罗地亚', 'Nigeria': '尼日利亚', 'Brazil': '巴西', 'Switzerland': '瑞士',
    'Costa Rica': '哥斯达黎加', 'Serbia': '塞尔维亚', 'Germany': '德国', 'Mexico': '墨西哥', 'Sweden': '瑞典', 'Korea Republic': '韩国',
    'Belgium': '比利时', 'Panama': '巴拿马', 'Tunisia': '突尼斯', 'England': '英格兰', 'Poland': '波兰', 'Senegal': '塞内加尔',
    'Colombia': '哥伦比亚', 'Japan': '日本'}
