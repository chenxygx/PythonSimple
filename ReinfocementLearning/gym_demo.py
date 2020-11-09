import gym

env = gym.make("CartPole-v0")
observation = env.reset()
env.render()
done = False
for _ in range(50000):
    env.render()
    action = env.action_space.sample()
    print(env.action_space)
    observation, reward, done, info = env.step(action)
    if done:
        observation = env.reset()
env.close()
