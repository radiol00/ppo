from ppo import PPO

ppo = PPO(input_shape=4,
            output_shape=2,
            hidden_shape=(64, 64,),
            actor_learning_rate=3e-4,
            critic_learning_rate=1e-3,
            epochs=20,
            clip=0.2,
            batch_size=5000)



def bang():
    ppo.calculate_advantage()
    print(ppo.advantages)
    ppo.clear_memory()

j = 0
for i in range(199):
    j += 1
    ppo.add_experience((1, 2, 3), 0, 1, False, 0, 0)
ppo.add_experience((1, 2, 3), 0, 200, True, 0, 0)

bang()

j = 0
for i in range(99):
    j += 1
    ppo.add_experience((1, 2, 3), 0, 1, False, 0, 0)
ppo.add_experience((1, 2, 3), 0, -200, True, 0, 0)

bang()

j = 0
for i in range(19):
    j += 1
    ppo.add_experience((1, 2, 3), 0, 1, False, 0, 0)
ppo.add_experience((1, 2, 3), 0, -200, True, 0, 0)

bang()