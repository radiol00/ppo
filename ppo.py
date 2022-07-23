import numpy as np
from keras.api._v2.keras.optimizers import Adam
from keras.api._v2.keras.layers import Input, Dense, Conv2D, Flatten
from keras.api._v2.keras import Model
import os
import time
import tensorflow as tf
from tensorflow_probability.python.distributions import Categorical
import scipy.signal


class PPO:
    def __init__(
        self,
        input_shape,
        output_shape,
        hidden_shape=(16,),
        name="PPO",
        discount_factor=0.99,
        lambda_val=0.97,
        batch_size=8,
        epochs=8,
        clip=0.2,
        actor_learning_rate=1e-5,
        critic_learning_rate=1e-4,
        activation="relu",
        target_kl=0.01,
        nn_class="dense"
    ):

        self.name = name
        self.nn_class = nn_class

        self.discount_factor = discount_factor
        self.lambda_val = lambda_val
        self.batch_size = batch_size
        self.epochs = epochs
        self.clip = clip
        self.target_kl = target_kl

        self.actor_optimizer = Adam(learning_rate=actor_learning_rate)
        self.critic_optimizer = Adam(learning_rate=critic_learning_rate)

        self.actor, self.critic = self.create_models(
            input_shape, hidden_shape, output_shape, activation)

        self.actor.summary()
        self.critic.summary()

        self.buffer = Buffer(lmbda=lambda_val, gamma=discount_factor)

    def create_models(self, input_shape, hidden_shape, output_shape, activation):
        if self.nn_class == "dense":
            return self.create_dense_models(input_shape, hidden_shape, output_shape, activation)
        elif self.nn_class == "conv":
            return self.create_conv_models(input_shape, hidden_shape, output_shape, activation)
        else:
            raise NotImplementedError

    def create_conv_models(self, input_shape, hidden_shape, output_shape, activation):
        input_layer = Input(shape=input_shape)
        prev_layer = input_layer

        for shape in hidden_shape:
            next_layer = Conv2D(
                shape, (3, 3), activation=activation)(prev_layer)
            prev_layer = next_layer

        flatten_layer = Flatten()(prev_layer)
        output_layer = Dense(output_shape, activation="softmax")(flatten_layer)

        actor = Model([input_layer], [output_layer], name="actor")

        input_layer = Input(shape=input_shape)
        prev_layer = input_layer

        for shape in hidden_shape:
            next_layer = Conv2D(
                shape, (3, 3), activation=activation)(prev_layer)
            prev_layer = next_layer

        flatten_layer = Flatten()(prev_layer)
        output_layer = Dense(1, activation=None)(flatten_layer)

        critic = Model([input_layer], [output_layer], name="critic")

        return actor, critic

    def create_dense_models(self, input_shape, hidden_shape, output_shape, activation):
        input_layer = Input(shape=input_shape)
        prev_layer = input_layer
        for shape in hidden_shape:
            next_layer = Dense(shape, activation=activation)(prev_layer)
            prev_layer = next_layer

        output_layer = Dense(output_shape, activation="softmax")(prev_layer)

        actor = Model([input_layer], [output_layer], name="actor")

        input_layer = Input(shape=input_shape)
        prev_layer = input_layer
        for shape in hidden_shape:
            next_layer = Dense(shape, activation=activation)(prev_layer)
            prev_layer = next_layer

        output_layer = Dense(1, activation=None)(prev_layer)

        critic = Model([input_layer], [output_layer], name="critic")

        return actor, critic

    def save_weights(self):
        path = os.path.join("weights", self.name)
        if not os.path.exists(path):
            os.makedirs(path)
        t = time.time()
        self.actor.save_weights(os.path.join(
            path, self.name + f"-{int(t)}-actor.hdf5"))
        self.critic.save_weights(os.path.join(
            path, self.name + f"-{int(t)}-critic.hdf5"))
        print(f"{self.name} weights saved")

    def load_weights(self, actor_path, critic_path):
        if actor_path is not None:
            self.actor.load_weights(actor_path)
            print(f"{self.name} actor weights loaded")
        if critic_path is not None:
            self.critic.load_weights(critic_path)
            print(f"{self.name} critic weights loaded")

    def policy(self, state):
        actor_pred = self.actor(
            np.array(state)[np.newaxis, :], training=False)[0]
        distribution = Categorical(probs=actor_pred)
        action = distribution.sample()
        return int(action), distribution.prob(action).numpy(), actor_pred

    def value(self, state):
        critic_pred = self.critic(np.array(state)[np.newaxis, :], training=False)[
            0][0].numpy()
        return critic_pred

    @tf.function
    def train_policy(self, states, actions, old_probs, advantages):
        with tf.GradientTape() as tape:
            new_probs = tf.clip_by_value(self.actor(
                states), clip_value_min=1e-30, clip_value_max=1e+30)
            new_probs = Categorical(probs=new_probs).prob(actions)

            prob_ratios = tf.divide(new_probs, old_probs)

            unclipped = tf.multiply(advantages, prob_ratios)
            clipped = tf.multiply(tf.clip_by_value(
                prob_ratios, 1-self.clip, 1+self.clip), advantages)

            policy_loss = tf.multiply(
                tf.constant(-1.0), tf.reduce_mean(tf.minimum(unclipped, clipped)))

        gradients = tape.gradient(policy_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(
            zip(gradients, self.actor.trainable_variables))

        kl = tf.reduce_mean(tf.divide(old_probs, new_probs))
        kl = tf.reduce_sum(kl)
        return kl

    @tf.function
    def train_value(self, states, returns):
        with tf.GradientTape() as tape:
            value_loss = tf.reduce_mean((returns - self.critic(states)) ** 2)
        gradients = tape.gradient(value_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(
            zip(gradients, self.critic.trainable_variables))

    def learn(self):
        rollout, size = self.buffer.get_rollout()
        batches = self.buffer.randomize_batches(self.batch_size, size)

        for batch in batches:
            traj = self.buffer.get_trajectory(rollout, batch)

            for _ in range(self.epochs):
                kl = self.train_policy(
                    traj["states"], traj["actions"], traj["probs"], traj["advantages"])
                if kl > 1.5 * self.target_kl:
                    break

            for _ in range(self.epochs):
                self.train_value(traj["states"], traj["returns"])


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
        advantage_mean, advantage_std = (
            np.mean(self.rollout["advantages"]), np.std(self.rollout["advantages"]),)
        self.rollout["norm_advantages"] = (
            self.rollout["advantages"] - advantage_mean) / advantage_std

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
        batches = [batch[i * batch_size:(i + 1) * batch_size]
                   for i in range((len(batch) + batch_size - 1) // batch_size)]
        return batches

    def finish_trajectory(self, value=0):
        rewards = np.array(self.memory["rewards"] + [value])
        values = np.array(self.memory["values"] + [value])
        self.memory["deltas"] = rewards[:-1] + \
            self.gamma * values[1:] - values[:-1]

        self.memory["advantages"] = self.dcs(
            self.memory["deltas"], self.gamma * self.lmbda)

        self.memory["returns"] = self.dcs(rewards, self.gamma)[:-1]

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
            trajectory[k] = tf.convert_to_tensor(
                [v[i] for i in batch], dtype=tf.float32)

        return trajectory
