from vizdoom import DoomGame, ScreenResolution
import random

class DoomEnv():
    def __init__(self):
        self.doom_game = DoomGame()  # 更正這裡，創建的應該是 DoomGame 實例
        self.config_path = "./config/basic.cfg"
        self.num_buttons = self.doom_game.get_available_buttons_size()  # 獲取按鈕數量
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
                

                
                


                reward = game.make_action([random.choice([1, 0]) for _ in range(game.get_available_buttons_size())])

                print("Reward:", reward)
                print("Game variables:", state.game_variables)

            print("Result:", game.get_total_reward())

        # 關閉遊戲
        game.close()



    

    # 定義左移動作
    def move_left(self):
        action = [0] * self.num_buttons
        action[self.doom_game.get_available_buttons().index('MOVE_LEFT')] = 1
        return action

    # 定義右移動作
    def move_right(self):
        action = [0] * self.num_buttons
        action[self.doom_game.get_available_buttons().index('MOVE_RIGHT')] = 1
        return action

    # 定義攻擊動作
    def attack(self):
        action = [0] * self.num_buttons
        action[self.doom_game.get_available_buttons().index('ATTACK')] = 1
        return action



if __name__ == "__main__":
    game_env = DoomEnv()
    game_env.step()
