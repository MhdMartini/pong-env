import numpy as np
import pygame as pg
from tqdm import tqdm
from itertools import count
from gym import spaces

W = 1
H = 2
SCALE = 400
W_WIN = W * SCALE
H_WIN = H * SCALE

V_BALL = 5
R_BALL = 0.02

H_PED = 0.05
W_PED = 0.2
R_H_PED = H_PED / 2
R_W_PED = W_PED / 2

Y_PED = 0.1

# indeces in state vector
I_X_PED = 0
I_X_BALL = 1
I_Y_BALL = 2
I_VX_BALL = 3
I_VY_BALL = 4

dt = 0.01

bg_color = pg.Color(20, 20, 20)
peddle_color = pg.Color(220, 220, 220)
ball_color = peddle_color
line_color = pg.Color(220, 220, 220)

FPS = 120


class PongSoloEnv:
    def __init__(self):
        self.n_states = 5
        actions = [-4, 0, 4]
        self.actions = actions

        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self.n_states,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)
        self.reward_range = range(-10, 11)
        self.metadata = None

        self.reset()

        self.screen = None

    def draw_state(self):
        self.screen.fill(bg_color)

        # draw peddle
        ped_x_corner = self.s[I_X_PED] - R_W_PED
        ped_y_corner = Y_PED - R_H_PED
        pg.draw.rect(self.screen,
                     peddle_color,
                     (ped_x_corner * SCALE, ped_y_corner * SCALE, W_PED * SCALE, H_PED * SCALE))

        # draw mid line
        pg.draw.line(self.screen, line_color,
                     (0, H_WIN / 2), (W_WIN, H_WIN / 2),
                     width=2)

        # draw ball
        pg.draw.circle(self.screen,
                       ball_color,
                       self.s[[I_X_BALL, I_Y_BALL]] * SCALE,
                       radius=R_BALL * SCALE)

    def init_s(self):
        # peddle position
        s = np.zeros(self.n_states)
        s[I_X_PED] = np.random.uniform(R_W_PED, W - R_W_PED)

        # ball position
        s[I_X_BALL] = np.random.uniform(R_BALL, W - R_BALL)
        s[I_Y_BALL] = H - R_BALL

        # ball velocity
        angle = 5 * np.pi / 6  # np.random.uniform(np.pi * 2)
        vx = V_BALL * np.sin(angle)
        vy = V_BALL * np.cos(angle)
        s[[I_VX_BALL, I_VY_BALL]] = [vx, vy]
        return s

    def step(self, a):
        r = 0
        terminal = False

        # handle peddle
        target_pos = self.s[I_X_PED] + self.actions[a] * dt
        if not R_W_PED <= target_pos <= W - R_W_PED:
            target_pos = max(min(W - R_W_PED, target_pos), R_W_PED)
        self.s[I_X_PED] = target_pos

        # detect y collision
        target_y = self.s[I_Y_BALL] + self.s[I_VY_BALL] * dt
        target_x = self.s[I_X_BALL] + self.s[I_VX_BALL] * dt
        if target_y - R_BALL - R_H_PED <= Y_PED:
            if abs(target_x - self.s[I_X_PED]) < R_W_PED:
                target_y = Y_PED + R_H_PED + R_BALL
                self.s[I_VY_BALL] *= -1
                self.s[I_VX_BALL] += a
                r = 10
            else:
                terminal = True
                r = -10

        elif target_y + R_BALL >= H:
            target_y = H - R_BALL
            self.s[I_VY_BALL] *= -1

        self.s[I_Y_BALL] = target_y

        # handle x collisiton
        if not R_BALL <= target_x <= W - R_BALL:
            target_x = max(min(W - R_BALL, target_x), R_BALL)
            self.s[I_VX_BALL] *= -1
            self.s[I_VX_BALL] += np.random.uniform(-0.01, 0.01)

        self.s[I_X_BALL] = target_x

        return np.copy(self.s), r, terminal, {}

    def init_pg(self):
        pg.init()
        self.clock = pg.time.Clock()
        screen = pg.display.set_mode((W_WIN, H_WIN))
        screen.fill(bg_color)
        return screen

    def render(self):
        if self.screen is None:
            self.screen = self.init_pg()
        if self.screen is False:
            return
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.screen = False
                pg.quit()
                return False

        self.clock.tick(FPS)
        self.draw_state()
        pg.display.flip()
        pg.display.update()
        return True

    def reset(self):
        self.s = self.init_s()
        self.screen = None
        return self.s

    def close(self):
        pg.quit()
