import numpy as np

class Game(object):
    def __init__(self):
        pass

    def run(self):
        pass

    def test(self):
        pass

    def run_epoch(self):
        pass

    def run_episode(self):
        pass


class DQNGame(Game):
    def __init__(self, max_epoch=200, steps_per_epoch=250000, max_start_ops=30, agent=None, memory=None, env=None,
                 final_epsilon=0.1, epsilon_decay=1000000):
        self.max_epoch = max_epoch
        self.steps_per_epoch = steps_per_epoch
        self.max_start_ops = max_start_ops
        self.steps_per_epoch = 250000
        self.current_step = 0
        self.current_epoch = 0
        self.agent = agent
        self.memory = memory
        self.env = env
        self.epsilon = 1.0
        self.final_epsilon = final_epsilon
        self.d_epsilon = (1.0 - final_epsilon) / epsilon_decay
        self.state = []
        self.file = open('log.txt', 'w')

    def run(self):
        for epoch in range(self.max_epoch):
            self.current_epoch = epoch
            self.run_epoch()

    def run_test(self):
        pass

    def run_epoch(self):
        while self.current_step < self.steps_per_epoch:
            self.run_episode()
        self.current_step = 0

    def run_episode(self):
        self.env.reset()
        frame = self.env.get_frame()
        self.state = [frame, frame, frame, frame]
        n = np.random.randint(1, self.max_start_ops)
        for _ in range(n-1):
            self.env.step(0)
            frame = self.env.get_frame()
            self.state.append(frame)
            self.state.pop(0)

        reward_sum = 0
        loss_sum = 0
        loss_cnt = 0
        qval_sum = 0
        qval_cnt = 0

        while not self.env.terminal:
            terminal = self.env.terminal
            self.state.append(frame)
            self.state.pop(0)
            if np.random.rand() < self.epsilon:
                act = np.random.randint(0, self.agent.num_actions)
            else:
                state = np.array(self.state)
                act, qval = self.agent.perceive(state)
                qval_sum += qval
                qval_cnt += 1

            reward = self.env.step(act)
            self.memory.add_sample(frame, act, reward, terminal)
            reward_sum += reward
            self.current_step += 1
            if self.memory.top > self.memory.start_sample_size:
                self.epsilon = max(self.final_epsilon, self.epsilon - self.d_epsilon)
                if self.current_step % 4 == 0:
                    states, actions, rewards, terminals, next_states = self.memory.sample_batch()
                    loss_sum += self.agent.update(states, actions, rewards, terminals, next_states)[0]
                    loss_cnt += 1
        info = "Epoch: %3d | Steps: %6d | Rewards: %5d | Mean Loss: %.3f | Mean Q: %.3f | Epsilon: %.3f" % \
               (self.current_epoch, self.current_step, reward_sum, loss_sum/(loss_cnt + 0.0001),
                qval_sum/(qval_cnt + 0.0001), self.epsilon)

        print info
        self.file.write(info + '\n')






