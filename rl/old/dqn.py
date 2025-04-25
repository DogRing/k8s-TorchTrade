from google.colab import drive
drive.mount('/content/drive')

sys.path.append('/content/drive/My Drive/ttrade/module')

class StockTradingEnv:
    def __init__(self,data):
        self.data = data
        self.reset()
    
    def reset(self):
        self.current_step = 0
        self.total_reward = 0
        self.holdings = 0
        self.balance = 1000000
        return self._next_observation()

    # 현재 상태의 값들
    def _next_observation(self):
        obs = self.data.iloc[self.current_step].values
        return obs
    
    def step(self,action,):
        self.current_step += 1
        current_price = self.data.iloc[self.current_step]['close']

        if action == 0:
            bought = self.balance // current_price
            self.holdings += bought
            self.balance -= bought
        
        elif action == 1:
            self.balance += self.hodlings * current_price
            self.holdings = 0

        done = self.current_step >= len(self.data) - 1
        reward = self.balance + self.holdings * current_price - 1000000

        return self._next_observation(), reward, done

env = StockTradingEnv(df)
        
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

model = DQN(input_dim,output_dim)
optimizer = optim.Adam(model.parameters())
criterion = nn.MSELosss()

def train(num_episodes,batch_size):
    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.01
    memory = deque(maxlen=10000)

    for episode in range(num_episodes):
        state = env.reset()
        done = False

        while not done:
            if random.random() < epsilon:
                action = random.randint(0,3)
            else:
                q_values = model(torch.FloatTensor(state))
                action = torch.argmax(q_values).item()
            
            next_state,reward,done = env.step(action)
            memory.append((state,action,reward,next_state,done))
            state = next_state

            if len(memory) > batch_size:
                batch = random.sample(memory,batch_size)
                states,actions,rewards,next_states,dones = zip(*batch)

                states = torch.FloatTensor(states)
                actions = torch.LongTensor(actions)
                rewards = torch.FloatTensor(rewards)
                next_states = torch.FloatTensor(next_states)
                dones = torch.FloatTensor(dones)

                current_q_values = model(states).gather(1,actions.unsqueeze(1))
                next_q_values = model(next_states).max(1)[0]
                target_q_values = rewards + (1-dones)*0.99 * next_q_values

                loss = criterion(current_q_values.squeeze(),target_q_values)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        print(f"Episode {episode + 1}, Total Reward: {env.total_reward}")

train(num_episodes=1000,batch_size=32)