from ppo import PPO

import numpy as np

ppo = PPO(input_shape=4,
            output_shape=2,
            hidden_shape=(64, 64,),
            actor_learning_rate=3e-4,
            critic_learning_rate=1e-3,
            epochs=20,
            clip=0.2,
            batch_size=8)


for i in range(4):
    ppo.buffer.add_experience((i, i, i), 0, 1, False, 0, 0)
ppo.buffer.add_experience((i+1, i+1, i+1), 0, 0, True, 0, 0)
ppo.buffer.finish()

for i in range(2):
    ppo.buffer.add_experience((i, i, i), 0, 1, False, 0, 0)
ppo.buffer.add_experience((i+1, i+1, i+1), 0, 0, True, 0, 0)
ppo.buffer.finish()

ppo.learn()

# j = 0
# for i in range(499):
#     j += 1
#     ppo.buffer.add_experience((1, 2, 3), 0, -1, False, 0, 0)
# ppo.buffer.add_experience((1, 2, 3), 0, 2_000, True, 0, 0)

# ppo.learn()