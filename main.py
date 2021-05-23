import gym
import random
import numpy as np
from time import sleep

env = gym.make('Taxi-v3').env
env.render()

# 0 = sul 1 = norte 2 = leste 3 = oeste 4 = pegar 5 = deixar
print(env.action_space)
print(env.observation_space)

from IPython.display import  clear_output

alpha = 0.1
gamma = 0.6
epsilon = 0.1

q_table = np.zeros([env.observation_space.n, env.action_space.n])

for i in range(100000):
    estado = env.reset()
    penalidades, recompensa = 0, 0
    done = False

    while not done:
        if random.uniform(0, 1) < epsilon:
            acao = env.action_space.sample()
        else:
            acao = np.argmax(q_table[estado])

        proximo_estado, recompensa, done, info = env.step(acao)

        q_antigo = q_table[estado, acao]
        proximo_maximo = np.max(q_table[proximo_estado])

        q_novo = (1 - alpha) * q_antigo + alpha * (recompensa + gamma * proximo_maximo)
        q_table[estado, acao] = q_novo

        if recompensa == -10:
            penalidades += 1

        estado = proximo_estado

print('=========================')
print('Treinamento concluido')

total_penalidades = 0
episodios = 50
frames = []

for _ in range(episodios):
    estado = env.reset()
    penalidades, recompensa = 0, 0
    done = False

    while not done:
        acao = np.argmax(q_table[estado])
        estado, recompensa, done, info = env.step(acao)

        if recompensa == -10:
            penalidades += 1

        frames.append({
            'frame': env.render(mode='ansi'),
            'state': estado,
            'action': acao,
            'reward': recompensa
        })
    total_penalidades += penalidades

print('Episódios', episodios)
print('Penalidades', total_penalidades)
print('=========================')

for frame in frames:
    clear_output(wait=True)
    print(frame['frame'])
    print('Estado', frame['state'])
    print('Ação', frame['action'])
    print('Recompensa', frame['reward'])
    sleep(.1)



