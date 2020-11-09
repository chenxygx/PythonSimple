import gym
import numpy as np
from forzen_lake_env import ForzenLakeEnv

# 构造走迷宫机器人
env = gym.make('FrozenLake-v0')
# 初始化Q表
Q = np.zeros([env.observation_space.n, env.action_space.n])
lr = 0.5  # 学习率 α
y = 0.95  # 贴现因子

# episode循环
success = 0
for i_episode in range(500000):
    s = env.reset()  # 重置环境
    for t in range(100000):  # step循环
        print(chr(27) + "[2J")
        env.render()  # 显示
        print('episode：', i_episode)
        print('success: ', success)
        print('rate:{:.2f}%'.format(success / max(1, i_episode) * 100, ))
        # ∑-greeedy策略
        a = np.argmax(Q[s, :] + np.random.randn(1, env.action_space.n) * max(1. / (i_episode + 1), 0.01))
        sl, r, d, info = env.step(a)
        # 更新Q表
        Q[s, a] = Q[s, a] + lr * (r + y * np.max(Q[sl, :]) - Q[s, a])
        s = sl
        if d:
            if r > 0.5:
                success += 1
            break
