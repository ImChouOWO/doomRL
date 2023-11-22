from vizdoom import DoomGame, ScreenResolution
import random

# 創建一個 DoomGame 實例
game = DoomGame()

# 載入配置
game.load_config("./config/basic.cfg")  # 這裡你需要指定配置文件的路徑

# 初始化遊戲
game.init()

# 遊戲主循環
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
