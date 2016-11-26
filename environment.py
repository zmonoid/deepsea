import numpy as np
from scipy.misc import imresize


class Environment(object):
    """
    Template Environment Object
    """

    def __init__(self):
        self.action_repeat = 0
        self.terminal = False
        self.display = False

    def reset(self):
        pass

    def step(self, action_index):
        for _ in range(self.action_repeat):
            pass
        return


class AtariEnvironment(Environment):
    """
    Atari Environment Object
    """

    def __init__(self, rom_path, action_repeat=4, death_end=True, width_resize=84, height_resize=84,
                 resize_mod='scale'):
        super(Environment, self).__init__()
        self.action_repeat = action_repeat
        self.death_end = death_end
        self.width_resize = width_resize
        self.height_resize = height_resize
        self.resize_mod = resize_mod
        self.display = False

        from ale_python_interface import ALEInterface
        self.ale = ALEInterface()
        self.ale.loadROM(rom_path)
        self.ale.setInt('random_seed', np.random.randint(1000))
        self.ale.setBool('display_screen', self.display)
        self.action_set = self.ale.getMinimalActionSet()
        self.num_actions = len(self.action_set)
        self.start_lives = self.ale.lives()
        width, height = self.ale.getScreenDims()
        self.currentScreen = np.empty((height, width), dtype=np.uint8)
        self.reset()

    def reset(self):
        self.ale.reset_game()
        self.ale.getScreenGrayscale(self.currentScreen)
        self.terminal = False

    def step(self, action, repeat=None):
        repeat = self.action_repeat if repeat is None else repeat
        reward = 0
        for _ in range(repeat):
            reward += self.ale.act(self.action_set[action])
        self.ale.getScreenGrayscale(self.currentScreen)
        self.terminal = self.death_end and self.ale.lives() < self.start_lives or self.ale.game_over()
        return reward

    def get_frame(self):
        if self.resize_mod == 'scale':
            return imresize(self.currentScreen, (self.width_resize, self.height_resize), interp='bilinear')
        elif self.resize_mod == 'crop':
            height, width = self.currentScreen.shape
            res = (height - width) / 2
            crop = self.currentScreen[res:(res + width), :]
            return imresize(crop, (self.width_resize, self.height_resize), interp='bilinear')


class GymEnvironment(Environment):
    def __init__(self):
        pass


class VrepEnvironment(Environment):
    def __init__(self):
        pass


class TorcsEnvironment(Environment):
    def __init__(self):
        pass


if __name__ == '__main__':
    env = AtariEnvironment(rom_path='./roms/breakout.bin', resize_mod='scale')
    import matplotlib.pyplot as plt

    frame = env.get_frame()
    plt.imshow(frame, cmap='gray')
    plt.show()
    print frame.shape
