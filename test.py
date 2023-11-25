from vizdoom import *
import random
import numpy as np
import time
class DoomEnv():
    def __init__(self):
        self.doom_game = DoomGame()  # 更正這裡，創建的應該是 DoomGame 實例
        self.config_path = "./config/basic.cfg"
        self.num_buttons = self.doom_game.get_available_buttons_size()  # 獲取按鈕數量
        self.action = [[1,0,0],[0,1,0],[0,0,1]]
    def step(self):
        game = self.doom_game
        game.load_config(self.config_path)  # 確保這裡使用的是 load_config 而不是 load
        game.init()
        episodes = 10
        for i in range(episodes):
            game.new_episode()
            while not game.is_episode_finished():
                # 獲取當前遊戲狀態和獎勵
                state = game.get_state()
                img = state.screen_buffer
                info = state.game_variables
                reward = game.make_action(random.choice(self.action),4)
                print(f"Epsiode: {i+1},Reward:", reward)
                time.sleep(0.02)
            print("Result:", game.get_total_reward())

        # 關閉遊戲
        game.close()
if __name__ == "__main__":
    game_env = DoomEnv()
    game_env.step()
