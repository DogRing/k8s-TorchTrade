class StockTradingEnv:
    def __init__(self,data):
        self.data = data
        self.current_step = 0
        self.max_steps = len()
        self.initial_balance = 1000000
        self.balance = self.initial_balance
        self.shares = 0
    
    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares = 0
        return sefl._get_observation()

    def step(self,action):
        self.current_step += 1
        current_price = self.data.iloc[self.current_step]['close']

        if action > 0:
            budget = int(self.balance * action / current_price)
            self.balance -= budget * current_price
            self.shares += budget
        elif action < 0:
            shares_to_sell = int()
            self.balance

