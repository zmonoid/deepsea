from environment import AtariEnvironment
from common import NumpyReplayMemory
from game import DQNGame
from agent import DQNAgent
import mxnet as mx

env = AtariEnvironment(rom_path='./roms/breakout.bin')
memory = NumpyReplayMemory()
agent = DQNAgent(num_actions=env.num_actions, ctx=mx.gpu(0))
game = DQNGame(agent=agent, env=env, memory=memory)
game.run()
