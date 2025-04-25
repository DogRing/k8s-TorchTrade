import torch
import random
from dataset import MultiTimeDataset as MTDataset

class TradingEnv:
    def __init__(self,path,ticks,input_dims,batch_size,max_step,device='cpu'):
        self.max_step = max_step
        self.datasets = [MTDataset(path,tick,input_dims,batch_size,device) for tick in ticks]
        self.reset()

    def reset(self,batch_index):
        self.dataset = self.datasets[random.randint(0,len(datasets)-1)]
        self.position = False
        self.current_price = None
        self.transaction_cost = 0.0005
        self.n_step = 0
        self.total_reward = 1
        self.last_reward = 0
        return dataset.get_batch(batch_index)

    def step(self,actions,labels,batch_index):
        rewards = np.zeros_like(actions, dtype=np.float32)
        done = False
        for i,(action, label) in enumerate(zip(actions,labels)):
            if action == 2 and not position:
                profit = price_change = (self.current_price - label) / label
                self.current_price = label
                self.position = True
            elif action == 0 and position:
                profit = (label - self.current_price) / label - self.transaction_cost
                self.current_price = label
                self.position = False
            else:
                if self.position:
                    profit = (label - self.current_price) / label * 0.01
                else:
                    profit = (self.current_price - label) / label * 0.01
            rewards[i] = profit
        self.n_step += 1
        self.total_reward *= np.prod(rewards + 1)
        if (self.total_reward < 0.9) or (self.n_step >= self.max_step):
            done = True
        rewards[0] += self.last_reward
        rewards = np.cumsum(rewards)
        self.last_reward = rewards[-1]
        states, labels = self.dataset.get_batch(batch_index)
        return states, labels, rewards, done

path = "/data/data/"
ticks = ["KRW-BTC"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 64
feature_dims = {
    '1': 4, '15':6, '60':6, '240':6, '1440':8
}

input_dims = {
    '1':140,'15':15,'60':15,'240':15,'1440':10
}

env = TradingEnv(path,ticks,input_dims,batch_size,device)
agent = PPOAgent(feature_dims, input_dims, env.n_actions, device)

update_interval = 2048
for episode in range(num_episodes):
    batch_index = 0
    state, label = env.reset(batch_index)
    episode_reward = 0
    
    for step in range(batch_index, batch_index+max_steps):
        action,log_prob,_,value = agent.model.get_action(state,deterministic=(random.random() < 0.5))
        next_batch_index = batch_index + 1
        next_state,next_label,reward,done,_ = env.step(action,label,next_batch_index)

        agent.memory.push(batch_index, reward, log_prob, value, done)
        state = next_state
        episode_reward += reward
        if (step + 1) % update_interval == 0:
            if np.any(done):
                next_value = 0
            else:
                _,_,_,next_value = agent.model.get_action(state, deterministic=True)
            agent.update(next_value, env.dataset)

        if done:
            break
    
    print(f"Episode: {episode+1}, Reward: {episode_reward}")