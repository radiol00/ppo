import numpy as np
import time
import gym
from ppo import PPO
env = gym.make("CartPole-v1", render_mode="human")
ppo = PPO(input_shape=4,
            output_shape=2,
            hidden_shape=(64, 64,),
            actor_learning_rate=3e-4,
            critic_learning_rate=1e-3,
            epochs=10,
            batch_size=100,
            name=f"PPO-{int(time.time())}")

ppo.load_weights("weights/PPO-1658512915/PPO-1658512915-1658514231-actor.hdf5", None)
episodes = 10
for _ in range(episodes):
    state, done, result = env.reset(seed=42), False, 0
    while not done:

        action, _, _ = ppo.policy(state)

        next_state, _, done, _ = env.step(action)

        state = next_state
        
env.close()
