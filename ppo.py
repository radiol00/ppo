from importlib.metadata import distribution
import numpy as np
from buffer import Buffer
from keras.api._v2.keras.optimizers import Adam
from keras.api._v2.keras.layers import Input, Dense
from keras.api._v2.keras import Model
import os
import time
import tensorflow as tf
from tensorflow_probability.python.distributions import Categorical

epsilon = np.finfo(np.float32).eps.item() 

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

        self.actor, self.critic = self.create_dense_models(input_shape, hidden_shape, output_shape, activation)

        self.actor.summary()
        self.critic.summary()

        self.buffer = Buffer(lmbda=lambda_val, gamma=discount_factor)


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

    def policy(self, state):
        actor_pred = self.actor(np.array(state)[np.newaxis, :], training=False)[0]
        distribution = Categorical(probs=actor_pred)
        action = distribution.sample()
        return int(action), distribution.prob(action), actor_pred

    def v(self, state):
        critic_pred = self.critic(np.array(state)[np.newaxis, :], training=False)[0][0].numpy()
        return critic_pred

    def train_policy(self, rollout):
        pass

    def train_value(self, rollout):
        pass

    def learn(self):
        # print(self.buffer.get_rollout())
        rollout, size = self.buffer.get_rollout()
        batches = self.buffer.randomize_batches(self.batch_size, size)
        print(batches)
        # for i in range(size):
        #     print(rollout['states'][i], rollout['rewards'][i], rollout['advantages'][i], rollout['returns'][i])
        # self.calculate_advantage()
        # batches = self.randomize_batches()
        # for batch in batches:
        #     states, actions, advantages, old_probs, old_vals = self.get_batch_buffer(batch)
        #     for _ in range(self.epochs):
        #         with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
        #             new_probs = self.actor(states)
        #             new_probs = tf.clip_by_value(new_probs, clip_value_min=1e-30, clip_value_max=1e+30)
        #             new_vals = self.critic(states)

        #             distributions = Categorical(probs=new_probs)

        #             new_probs = distributions.prob(actions)
        #             prob_ratios = tf.divide(new_probs, old_probs)

        #             unclipped = tf.multiply(advantages, prob_ratios)
        #             clipped = tf.multiply(tf.clip_by_value(prob_ratios, 1-self.clip, 1+self.clip), advantages)
        #             surrogate_obj = tf.multiply(tf.constant(-1.0), tf.reduce_mean(tf.minimum(unclipped, clipped)))

        #             loss = tf.subtract(tf.add(advantages, old_vals), new_vals)
        #             loss = tf.square(loss)
        #             loss = tf.reduce_mean(loss)

        #         grad1 = tape1.gradient(surrogate_obj, self.actor.trainable_variables)
        #         grad2 = tape2.gradient(loss, self.critic.trainable_variables)
        #         self.actor_optimizer.apply_gradients(zip(grad1, self.actor.trainable_variables))
        #         self.critic_optimizer.apply_gradients(zip(grad2, self.critic.trainable_variables))

        # self.clear_memory()