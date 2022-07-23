import numpy as np
from buffer import Buffer
from keras.api._v2.keras.optimizers import Adam
from keras.api._v2.keras.layers import Input, Dense, Conv2D, Flatten
from keras.api._v2.keras import Model
import os
import time
import tensorflow as tf
from tensorflow_probability.python.distributions import Categorical


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
                shape, (2, 2), activation=activation)(prev_layer)
            prev_layer = next_layer

        flatten_layer = Flatten()(prev_layer)
        output_layer = Dense(output_shape, activation="softmax")(flatten_layer)

        actor = Model([input_layer], [output_layer], name="actor")

        input_layer = Input(shape=input_shape)
        prev_layer = input_layer

        for shape in hidden_shape:
            next_layer = Conv2D(
                shape, (2, 2), activation=activation)(prev_layer)
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
