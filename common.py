import numpy as np
import mxnet as mx


class ReplayMemory(object):
    def __init__(self):
        pass

    def add_sample(self):
        pass

    def random_batch(self):
        pass


class NumpyReplayMemory(ReplayMemory):
    def __init__(self, frame_shape=(84, 84), size=1000000, start_sample_size=50000, frame_scale=255.0):
        super(ReplayMemory, self).__init__()
        width, height = frame_shape
        self.frame_scale = frame_scale
        self.size_ = size
        self.start_sample_size = start_sample_size
        self.size = self.size_ + self.start_sample_size
        self.frames = np.zeros((self.size, width, height), dtype='float32')
        self.actions = np.zeros(self.size, dtype='float32')
        self.rewards = np.zeros(self.size, dtype='float32')
        self.terminals = np.zeros(self.size, dtype='float32')

        self.current = 0
        self.top = 0

    def add_sample(self, frame, action, reward, terminal):
        assert self.current < self.size
        assert self.top < self.size
        assert self.current <= self.top

        self.frames[self.current, :] = frame / self.frame_scale
        self.actions[self.current] = action
        self.rewards[self.current] = reward
        self.terminals[self.current] = terminal

        if self.current == self.top:
            self.current += 1
            self.top += 1
        elif self.current < self.top:
            self.current += 1

        if self.current < self.size_:
            return
        else:
            if terminal:
                self.top = self.current
                self.current = 0

    def sample_batch(self, input_shape=(32, 4, 84, 84)):
        assert self.top > self.start_sample_size
        batch_size, n_frames, width, height = input_shape

        states = np.zeros(input_shape, dtype='float32')
        next_states = np.zeros(input_shape, dtype='float32')
        actions = np.zeros(batch_size, dtype='float32')
        rewards = np.zeros(batch_size, dtype='float32')
        terminals = np.zeros(batch_size, dtype='float32')

        count = 0
        indexs = []
        while count < batch_size:
            idx = np.random.randint(4, self.top-2)
            if np.any(self.terminals[idx-3:idx+1]) or idx in indexs or 0 < idx - self.current < 4:
                continue
            else:
                states[count, :] = self.frames[idx-3:idx+1, :]
                next_states[count, :] = self.frames[idx-2:idx+2, :]
                actions[count] = self.actions[idx]
                rewards[count] = self.rewards[idx]
                terminals[count] = self.terminals[idx]
                count += 1
                indexs.append(idx)
        return states, actions, rewards, terminals, next_states


if __name__ == '__main__':
    memory = NumpyReplayMemory(size=1000, start_sample_size=100)
    for i in xrange(3000):
        frame = np.random.randn(*(84, 84))
        action = np.random.randint(0, 10)
        reward = np.random.randint(0, 8)
        terminal = np.random.randint(0, 2)
        memory.add_sample(frame, action, reward, terminal)

    for _ in xrange(100):
        memory.sample_batch()







