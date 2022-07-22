import numpy as np
import time
import gym
from ppo import PPO
env = gym.make("CartPole-v1")
ppo = PPO(input_shape=4,
            output_shape=2,
            hidden_shape=(64, 64,),
            actor_learning_rate=1e-4,
            critic_learning_rate=1e-3,
            epochs=80,
            batch_size=100,
            name=f"PPO-{int(time.time())}")

steps_per_session = 5_000
sessions = 25
for i in range(sessions):
    session_results, steps = [], 0
    while steps < steps_per_session:
        state, done, result = env.reset(seed=42), False, 0
        while not done:

            action, action_prob, probs = ppo.policy(state)
            value = ppo.value(state)

            next_state, reward, done, _ = env.step(action)
            result += reward

            ppo.buffer.add_experience(state, action, reward, done, action_prob, value)

            state = next_state

            steps += 1

            if done or steps == steps_per_session:
                ppo.buffer.finish_trajectory(0 if done else ppo.value(state))
                session_results.append(result)
                done = True
    print(f"Session {i+1}, Mean reward: {np.mean(session_results):.0f}")
    ppo.learn()
        
env.close()
ppo.save_weights()
