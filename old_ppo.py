from importlib.metadata import distribution
import numpy as np
from keras.api._v2.keras.optimizers import Adam
from keras.api._v2.keras.layers import Input, Dense
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
        activation="tanh"
        ):

        self.name = name
        self.discount_factor = discount_factor
        self.lambda_val = lambda_val
        self.batch_size = batch_size
        self.epochs = epochs
        self.clip = clip

        self.actor_optimizer = Adam(learning_rate=actor_learning_rate)
        self.critic_optimizer = Adam(learning_rate=critic_learning_rate)

        self.actor, self.critic = self.create_dense_relu_models(input_shape, hidden_shape, output_shape, activation)

        self.actor.summary()
        self.critic.summary()

        self.clear_memory()


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
        self.actor.save_weights(os.path.join(path, self.name + f"-{int(t)}-actor.hdf5"))
        self.critic.save_weights(os.path.join(path, self.name + f"-{int(t)}-critic.hdf5"))
        print(f"{self.name} weights saved")

    def load_weights(self, actor_path, critic_path):
        self.actor.load_weights(actor_path)
        print(f"{self.name} actor weights loaded")
        self.critic.load_weights(critic_path)
        print(f"{self.name} critic weights loaded")


    def clear_memory(self):
        self.states = []
        self.rewards = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.advantages = []
        self.dones = []

    def add_experience(self, state, action, reward, done, prob, val):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.probs.append(prob)
        self.vals.append(val)
        self.dones.append(done)

    def randomize_batches(self):
        memories = len(self.states)
        size = self.batch_size
        batch = np.arange(memories)
        np.random.shuffle(batch)
        batches = [batch[i * size:(i + 1) * size] for i in range((len(batch) + size - 1) // size)]
        return batches

    def calculate_advantage(self):
        self.advantages = []
        advantages = []

        rewards = np.array(self.rewards)

        for j in range(len(rewards)):
            advantage = 0
            reduction = 1

            for i in range(j, len(rewards) - 1):
                if self.dones[i]:
                    break
                reward = rewards[i]
                val = self.vals[i]
                next_val = self.vals[i+1]
                advantage += reduction * ((reward + self.discount_factor * next_val) - val)
                reduction = reduction * self.discount_factor * self.lambda_val

            if j == len(rewards) - 1 or self.dones[j]:
                advantage = rewards[j]

            advantages.append(advantage)

            if j == len(rewards) - 1 or self.dones[j]:
                advantages = np.array(advantages)
                # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)
                self.advantages.extend(advantages)
                advantages = []


        self.advantages = np.array(self.advantages)
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-10)

    def get_batch_buffer(self, batch):
        batch_buffer = {}

        states = [self.states[i] for i in batch]
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        batch_buffer["states"] = states

        vals = [self.vals[i] for i in batch]
        vals = tf.convert_to_tensor(vals, dtype=tf.float32)
        batch_buffer["vals"] = vals

        actions = [self.actions[i] for i in batch]
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        batch_buffer["actions"] = actions

        probs = [self.probs[i] for i in batch]
        probs = tf.convert_to_tensor(probs, dtype=tf.float32)
        batch_buffer["probs"] = probs

        return batch_buffer


    def predict(self, state):
        actor_pred = self.actor(np.array(state)[np.newaxis, :], training=False)[0]
        critic_pred = self.critic(np.array(state)[np.newaxis, :], training=False)[0][0].numpy()

        distribution = Categorical(probs=actor_pred)
        action = distribution.sample()

        return int(action), distribution.prob(action), actor_pred, critic_pred

    def learn(self):
        self.calculate_advantage()
        batches = self.randomize_batches()
        for batch in batches:
            states, actions, advantages, old_probs, old_vals = self.get_batch_buffer(batch)
            for i in range(self.epochs):
                with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
                    new_probs = self.actor(states)
                    new_probs = tf.clip_by_value(new_probs, clip_value_min=1e-30, clip_value_max=1e+30)
                    new_vals = self.critic(states)

                    distributions = Categorical(probs=new_probs)

                    new_probs = distributions.prob(actions)
                    prob_ratios = tf.divide(new_probs, old_probs)

                    unclipped = tf.multiply(advantages, prob_ratios)
                    clipped = tf.multiply(tf.clip_by_value(prob_ratios, 1-self.clip, 1+self.clip), advantages)
                    surrogate_obj = tf.multiply(tf.constant(-1.0), tf.reduce_mean(tf.minimum(unclipped, clipped)))

                    loss = tf.subtract(tf.add(advantages, old_vals), new_vals)
                    loss = tf.square(loss)
                    loss = tf.reduce_mean(loss)

                grad1 = tape1.gradient(surrogate_obj, self.actor.trainable_variables)
                grad2 = tape2.gradient(loss, self.critic.trainable_variables)
                self.actor_optimizer.apply_gradients(zip(grad1, self.actor.trainable_variables))
                self.critic_optimizer.apply_gradients(zip(grad2, self.critic.trainable_variables))

        self.clear_memory()



