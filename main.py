from vizdoom import *
import random
import numpy as np
import gym
from gym import spaces
import cv2
class DoomEnv(gym.Env):
    def __init__(self):
        super(DoomEnv, self).__init__()

        self.doom_game = DoomGame()
        self.config_path = "./config/basic.cfg"
        self.doom_game.load_config(self.config_path)
        self.doom_game.init()

        self.action_space = spaces.Discrete(3)  # 假設有三種動作
        self.observation_space = spaces.Box(low=0, high=255, shape=(240, 320, 3), dtype=np.uint8)  # 假設螢幕解析度為 320x240
        self.resize = [160,100]
    def step(self, action):
        actions = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        reward = self.doom_game.make_action(actions[action], 4)
        state = self.doom_game.get_state()
        done = self.doom_game.is_episode_finished()
        info = {}
        if state is not None:
            img = state.screen_buffer
            img =self.img_preprocess(img)
            info = state.game_variables
        else:
            img = np.zeros((self.resize[1], self.resize[0], 1), dtype=np.uint8)
        return img, reward, done, info

    def reset(self):
        self.doom_game.new_episode()
        state = self.doom_game.get_state()
        if state is not None:
            img = state.screen_buffer
            img = self.img_preprocess(img)  # 同樣，處理初始圖像
        else:
            img = np.zeros((self.resize[1], self.resize[0], 1), dtype=np.uint8)
        return img

    
    def img_preprocess(self, game_state):
        # game_state格式轉換為（高度 x 寬度 x 通道數）
        if game_state.ndim == 3 and game_state.shape[0] == 3:
            game_state = game_state.transpose(1, 2, 0)
            gray = cv2.cvtColor(game_state, cv2.COLOR_RGB2GRAY)
        elif game_state.ndim == 2 or (game_state.ndim == 3 and game_state.shape[0] == 1):
            gray = game_state.squeeze(0)  # 如果已經是單通道，去除多餘的維度
        else:
            raise ValueError("Unexpected game state shape: {}".format(game_state.shape))

        resized = cv2.resize(gray, (self.resize[0], self.resize[1]), interpolation=cv2.INTER_CUBIC)
        state = np.reshape(resized, (self.resize[1], self.resize[0], 1))
        return state





    def render(self, mode='human'):
        pass  # 可以根據需要實現渲染邏輯

    def close(self):
        self.doom_game.close()

if __name__ == "__main__":
    env = DoomEnv()
    episodes = 10
    for i in range(episodes):
        env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            state, reward, done, info = env.step(action)
            print(f"Episode: {i+1}, Reward: {reward}")
        print("Result:", env.doom_game.get_total_reward())
    env.close()
