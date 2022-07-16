import gym
from ppo import PPO
from matplotlib import pyplot as plt
# env = gym.make("CartPole-v1", render_mode='human')
env = gym.make("CartPole-v1")
import time
import numpy as np
per_steps = 10_000
ppo = PPO(input_shape=4,
            output_shape=2,
            hidden_shape=(64, 64,),
            actor_learning_rate=1e-4,
            critic_learning_rate=1e-3,
            epochs=20,
            clip=0.2,
            batch_size=per_steps//10,
            name=f"PPO-{int(time.time())}")
steps = 0

essa = []
while True:
    done = False
    episode_reward = 0
    state, info = env.reset(seed=42, return_info=True)
    
    while not done:
        steps = steps + 1
        action, prob, probs, val = ppo.predict(state)
        next_state, reward, done, _ = env.step(action)
        state = next_state
        episode_reward += reward
        
        ppo.add_experience(state, action, -1 if not done else 2_000, done, prob, val)

        if done:
            essa.append(episode_reward)
        
        if steps % per_steps == 0 and steps != 0:
            essa = np.array(essa)
            print("mean:", int(essa.mean()), "max:", essa.max(), "min:", essa.min())
            steps = 0
            essa = []
            ppo.learn()
            ppo.save_weights()
        
env.close()
ppo.save_weights()
plt.show()
