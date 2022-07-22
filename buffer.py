import numpy as np
import tensorflow as tf
import scipy.signal

class Buffer:

    def __init__(self, lmbda, gamma):
        self.clear_memory()
        self.clear_rollout()
        self.lmbda = lmbda
        self.gamma = gamma
        self.memories = 0

    def keys(self):
        return ["states", "rewards", "actions", "probs", "values", "dones",
                "advantages", "norm_advantages", "deltas", "returns"]

    def clear_rollout(self):
        self.rollout = {}
        for key in self.keys():
            self.rollout[key] = []

    def clear_memory(self):
        self.memory = {}
        for key in self.keys():
            self.memory[key] = []

    def push_memories_to_rollout(self):
        for key in self.keys():
            self.rollout[key].extend(self.memory[key])
    
    def normalize_advantage_in_rollout(self):
        advantage_mean, advantage_std = (np.mean(self.rollout["advantages"]), np.std(self.rollout["advantages"]),)
        self.rollout["norm_advantages"]  = (self.rollout["advantages"] - advantage_mean) / advantage_std

    def add_experience(self, state, action, reward, done, prob, value):
        self.memory["states"].append(state)
        self.memory["rewards"].append(reward)
        self.memory["actions"].append(action)
        self.memory["probs"].append(prob)
        self.memory["values"].append(value)
        self.memory["dones"].append(int(done))
        self.memories += 1

    def randomize_batches(self, batch_size, buffer_size):
        batch = np.arange(buffer_size)
        np.random.shuffle(batch)
        batches = [batch[i * batch_size:(i + 1) * batch_size] for i in range((len(batch) + batch_size - 1) // batch_size)]
        return batches

    def finish_trajectory(self, value=0):
        rewards = np.array(self.memory["rewards"] + [value])
        values = np.array(self.memory["values"] + [value])
        self.memory["deltas"] = rewards[:-1] + self.gamma * values[1:] - values[:-1]

        self.memory["advantages"] = self.dcs(self.memory["deltas"], self.gamma * self.lmbda)

        self.memory["returns"]  = self.dcs(rewards, self.gamma)[:-1]

        self.push_memories_to_rollout()
        self.clear_memory()


    def dcs(self, x, discount):
        return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

    def get_rollout(self):
        self.normalize_advantage_in_rollout()

        rollout = {}
        for key in self.keys():
            rollout[key] = np.array(self.rollout[key])
            
        self.clear_rollout()
        memories, self.memories = self.memories, 0
        return rollout, memories

    def get_trajectory(self, rollout, batch):
        trajectory = {}

        for k, v in rollout.items():
            trajectory[k] = tf.convert_to_tensor([v[i] for i in batch], dtype=tf.float32)

        return trajectory