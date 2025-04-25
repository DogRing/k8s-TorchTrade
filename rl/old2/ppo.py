import torch
from model import EnhancedMultiTimeframeModel as MTModel
from dataset import MultiTimeDataset as MTDataset

GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPSILON = 0.2
CRITIC_DISCOUNT = 0.5
ENTROPY_BETA = 0.01
LEARNING_RATE = 3e-4
PPO_EPOCHS = 10
MINI_BATCH_SIZE = 64
MAX_GRAD_NORM = 0.5

class PPOMemory:
    def __init__(self):
        self.states_index = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    def push(self,state_index,action,reward,value,log_prob,done):
        self.states_index.append(state_index)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    def get(self):
        return (
            self.states_index,
            self.actions,
            self.rewards,
            self.values,
            self.log_probs,
            self.dones
        )
    def clear(self):
        self.states_index.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()

class PPOAgent:
    def __init__(self, feature_dims, input_dims, n_actions, device):
        self.policy = MTModel(feature_dims, input_dims, n_actions).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=LEARNING_RATE)
        self.memory = PPOMemory()
        self.mode = "last"
        self.device = device
    def compute_gae(self, next_value, rewards, values, dones):
        values = values + [[next_value]]
        gae = 0
        returns = []
        advantages = []
        for steps in reversed(range(len(rewards))):
            returns_batch = []
            advantages_batch = []
            for step in reversed(range(len(rewards[steps]))):
                delta = rewards[steps][step] + GAMMA * values[steps + 1][0] * (1 - dones[steps][step]) - values[steps][step]
                gae = delta + GAMMA * GAE_LAMBDA * (1 - dones[steps][step]) * gae
                returns_batch.insert(0, gae + values[steps][step])
                advantages_batch.insert(0, gae)
            returns.append(returns_batch)
            advantages.append(advantages_batch)
        return returns, advantages
    def update(self,next_value,dataset):
        states_indexs, actions, rewards, values, old_log_probs, dones = self.memory.get()
        returns, advantages = self.comput_gae(next_value,rewards,values,dones)
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(PPO_EPOCHS):
            for batch_idx in states_indexs:
                mb_states = dataset.get_batch(batch_idx,self.mode)
                mb_actions = actions[batch_idx]
                mb_returns = returns[batch_idx]
                mb_advantages = advantages[batch_idx]
                mb_old_log_probs = old_log_probs[batch_idx]
                
                new_log_prob,entropy,values = self.policy.get_logprob(mb_states,mb_actions,self.mode)

                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - CLIP_EPSILON, 1.0 + CLIP_EPSILON) * mb_advantages
                actor_loss = -torch.min(surr1,surr2).mean()

                critic_loss = F.mse_loss(values.squeeze(-1), mb_returns)
                loss = actor_loss + CRITIC_DISCOUNT * critic_loss - ENTROPY_BETA * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(),MAX_GRAD_NORM)
                self.optimizer.step()
        self.memory.clear()