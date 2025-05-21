import random
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import argparse
import wandb
from tqdm import tqdm

import time

def init_layer_uniform(layer: nn.Linear, init_w: float = 3e-3) -> nn.Linear:
    """Init uniform parameters on the single layer."""
    layer.weight.data.uniform_(-init_w, init_w)
    layer.bias.data.uniform_(-init_w, init_w)
    return layer

class Actor(nn.Module):
    def __init__(
        self,
        beta: float,
        gamma: float,
        dim_observation: int,
        dim_action: int,
        log_std_min: float = -20,
        log_std_max: float = 2,
        hidden_dim: int = 128,
    ):
        """Initialize.You can design your own Actor architecture."""
        super(Actor, self).__init__()

        self.beta = beta  # entropy regularization parameter
        self.gamma = gamma  # shrinking factor

        # set the log std range
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # set the hidden layers
        self.hidden1 = nn.Linear(dim_observation, hidden_dim)
        self.hidden2 = nn.Linear(hidden_dim, hidden_dim)

        # set log_std layer
        self.log_std_layer = nn.Linear(hidden_dim, dim_action)
        self.log_std_layer = init_layer_uniform(self.log_std_layer)

        # set mean layer
        self.mu_layer = nn.Linear(hidden_dim, dim_action)
        self.mu_layer = init_layer_uniform(self.mu_layer)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        x = F.relu(self.hidden1(state))
        x = F.relu(self.hidden2(x))

        # get mean
        mu = self.mu_layer(x).tanh()

        # get std
        log_std = self.log_std_layer(x).tanh()
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1)
        std = torch.exp(log_std)

        # sample actions
        dist = Normal(mu, std)
        entropy = dist.entropy().sum(dim=-1)
        z = dist.rsample()

        # normalize action and log_prob
        # see appendix C of [2]
        action = z.tanh()
        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-7)
        log_prob = log_prob.sum(-1, keepdim=True)

        return action, log_prob, entropy, mu, std


class CriticQ(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 128):
        """Initialize. You can design your own Q Critic architecture."""
        super(CriticQ, self).__init__()

        self.hidden1 = nn.Linear(in_dim, hidden_dim)
        self.hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)
        self.out = init_layer_uniform(self.out)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        x = torch.cat((state, action), dim=-1)
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        value = self.out(x)

        return value


class CriticV(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 128):
        """Initialize. You can design your own V Critic architecture."""
        super(CriticV, self).__init__()

        self.hidden1 = nn.Linear(in_dim, hidden_dim)
        self.hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)
        self.out = init_layer_uniform(self.out)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        x = F.relu(self.hidden1(state))
        x = F.relu(self.hidden2(x))
        value = self.out(x)

        return value
    
class ReplayBuffer:
    """A simple numpy replay buffer."""

    def __init__(self, obs_dim: int, act_dim: int, size: int, batch_size: int = 32):
        """Initializate."""
        self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.next_obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros([size], dtype=np.float32)
        self.done_buf = np.zeros([size], dtype=np.float32)
        self.max_size, self.batch_size = size, batch_size
        self.ptr, self.size = 0, 0

    def store(
        self,
        obs: np.ndarray,
        act: np.ndarray,
        rew: float,
        next_obs: np.ndarray,
        done: bool,
    ):
        """Store the transition in buffer."""
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self) -> Dict[str, np.ndarray]:
        """Randomly sample a batch of experiences from memory."""
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)
        return dict(
            obs=self.obs_buf[idxs],
            next_obs=self.next_obs_buf[idxs],
            acts=self.acts_buf[idxs],
            rews=self.rews_buf[idxs],
            done=self.done_buf[idxs],
        )

    def __len__(self) -> int:
        return self.size


class SACAgent:
    """SAC agent interacting with environment. """

    def __init__(self, env: gym.Env, args=None):
        """Initialize."""
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        self.env = env
        self.memory_size = args.memory_size
        self.batch_size = args.batch_size        
        self.memory = ReplayBuffer(obs_dim, action_dim, self.memory_size, self.batch_size)
        self.gamma = args.discount_factor
        self.tau = args.tau
        self.lr = args.lr
        self.initial_random_steps = args.initial_random_steps
        self.policy_update_freq = args.policy_update_freq
        self.seed = args.seed
        self.num_steps = args.num_steps
        
        # device: cpu / gpu
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # automatic entropy tuning
        self.target_entropy = -np.prod((action_dim,)).item()  # heuristic
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.lr)

        # actor
        self.actor = Actor(obs_dim, action_dim).to(self.device)

        # v function
        self.vf = CriticV(obs_dim).to(self.device)
        self.vf_target = CriticV(obs_dim).to(self.device)
        self.vf_target.load_state_dict(self.vf.state_dict())

        # q function
        self.qf_1 = CriticQ(obs_dim + action_dim).to(self.device)
        self.qf_2 = CriticQ(obs_dim + action_dim).to(self.device)

        # optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.vf_optimizer = optim.Adam(self.vf.parameters(), lr=self.lr)
        self.qf_1_optimizer = optim.Adam(self.qf_1.parameters(), lr=self.lr)
        self.qf_2_optimizer = optim.Adam(self.qf_2.parameters(), lr=self.lr)

        # transition to store in memory
        self.transition = list()

        # total steps count
        self.total_step = 0

        # mode: train / test
        self.is_test = False

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input state."""
        # if initial random action should be conducted
        if self.total_step < self.initial_random_steps and not self.is_test:
            selected_action = self.env.action_space.sample()
        else:
            selected_action = (
                self.actor(torch.FloatTensor(state).to(self.device))[0].detach().cpu().numpy()
            )

        self.transition = [state, selected_action]

        return selected_action

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool]:
        """Take an action and return the response of the env."""
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated

        if not self.is_test:
            self.transition += [reward, next_state, done]
            self.memory.store(*self.transition)

        return next_state, reward, done

    def update_model(self) -> Tuple[torch.Tensor, ...]:
        """Update the model by stochastic gradient descent."""
        device = self.device  # for shortening the following lines

        samples = self.memory.sample_batch()
        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.FloatTensor(samples["acts"]).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)
        new_action, log_prob, _, _, _ = self.actor(state)

        # train alpha (dual problem)
        alpha_loss = (-self.log_alpha.exp() * (log_prob + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        alpha = self.log_alpha.exp()  # used for the actor loss calculation

        # Computing q function loss
        mask = 1 - done
        
        ########## Your Code (5-10 lines)##########

        with torch.no_grad():
            target_v = self.vf_target(next_state)
            y = reward + self.gamma * mask * target_v

        q1_pred = self.qf_1(state, action)
        q2_pred = self.qf_2(state, action)

        qf_1_loss = F.mse_loss(q1_pred, y)
        qf_2_loss = F.mse_loss(q2_pred, y)

        ########## End of Your Code ##########
        
        
        # Computing v function loss
        ########## Your Code (~5 lines)##########

        q1_new = self.qf_1(state, new_action)
        q2_new = self.qf_2(state, new_action)
        min_q_new = torch.min(q1_new, q2_new)

        vf_pred = self.vf(state)
        vf_target = min_q_new - alpha * log_prob
        vf_loss = F.mse_loss(vf_pred, vf_target.detach())

        ########## End of Your Code ##########


        if self.total_step % self.policy_update_freq == 0:
            # Computing actor loss
            ########## Your Code (<5 lines)##########

            actor_loss = (alpha * log_prob - min_q_new).mean()

            ########## End of Your Code ##########

            # train actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # target update (vf)
            self._target_soft_update()
        else:
            actor_loss = torch.zeros(())

        # train Q functions
        self.qf_1_optimizer.zero_grad()
        qf_1_loss.backward()
        self.qf_1_optimizer.step()

        self.qf_2_optimizer.zero_grad()
        qf_2_loss.backward()
        self.qf_2_optimizer.step()

        qf_loss = qf_1_loss + qf_2_loss

        # train V function
        self.vf_optimizer.zero_grad()
        vf_loss.backward()
        self.vf_optimizer.step()

        return actor_loss.data, qf_loss.data, vf_loss.data, alpha_loss.data

    def train(self):
        """Train the agent."""
        self.is_test = False

        state, _ = self.env.reset(seed=self.seed)
        actor_losses, qf_losses, vf_losses, alpha_losses = [], [], [], []
        scores = []
        score = 0
        ep = 0
        ewma_reward = 0
        for self.total_step in tqdm(range(1, self.num_steps + 1)):
            action = self.select_action(state)
            next_state, reward, done = self.step(action)

            state = next_state
            score += reward
            actor_loss = 0
            qf_loss  = 0
            vf_loss = 0
            alpha_loss = 0
            
            # if episode ends
            if done:
                scores.append(score)
                ewma_reward = 0.05 * score + (1 - 0.05) * ewma_reward
                ep += 1
                
                with torch.no_grad():
                    self.test(video_folder="video/halfcheetah")
                self.is_test = False
                
                print(f"Episode {ep} (Total step = {self.total_step}): Total Reward = {self.score}. Test EWMA = {self.ewma_reward}")
                # W&B logging
                wandb.log({
                    "episode": ep,
                    "return": score,
                    "ewma return": ewma_reward,
                    "test/score": self.score,
                    "test/ewma return": self.ewma_reward,
                    }) 
                
                score = 0
                state, _ = self.env.reset(seed=self.seed)
                
            # if training is ready
            if len(self.memory) >= self.batch_size and self.total_step > self.initial_random_steps:
                losses = self.update_model()
                actor_loss = losses[0].cpu().numpy()
                qf_loss = losses[1].cpu().numpy()                                                
                vf_loss = losses[2].cpu().numpy()
                alpha_loss = losses[3].cpu().numpy()
                actor_losses.append(actor_loss)
                qf_losses.append(qf_loss)
                vf_losses.append(vf_loss)
                alpha_losses.append(alpha_loss)

            # W&B logging
            wandb.log({
                "step": self.total_step,
                "actor loss": actor_loss,
                "q loss": qf_loss,
                "v loss": vf_loss,
                "alpha loss": alpha_loss
                }) 

        self.env.close()

    def test(self, video_folder: str, save_video: bool = False):
        """Test the agent."""
        self.is_test = True

        tmp_env = self.env

        # the recording doesn't work
        # if save_video:
        #     self.env = gym.wrappers.RecordVideo(self.env, video_folder=video_folder)

        state, _ = self.env.reset(seed=self.seed)
        done = False
        score = 0

        while not done:
            action = self.select_action(state)
            next_state, reward, done = self.step(action)

            state = next_state
            score += reward

        # Update ewma_reward after the episode
        if not hasattr(self, "ewma_reward"):
            self.ewma_reward = score
            self.score = score
        else:
            self.ewma_reward = 0.05 * score + (1 - 0.05) * self.ewma_reward
            self.score = score

        # print("score: ", score)
        self.env.close()

        self.env = tmp_env

    def _target_soft_update(self):
        """Soft-update: target = tau*local + (1-tau)*target."""
        tau = self.tau

        for t_param, l_param in zip(self.vf_target.parameters(), self.vf.parameters()):
            t_param.data.copy_(tau * l_param.data + (1.0 - tau) * t_param.data)

class ActionNormalizer(gym.ActionWrapper):
    """Rescale and relocate the actions."""

    def action(self, action: np.ndarray) -> np.ndarray:
        """Change the range (-1, 1) to (low, high)."""
        low = self.action_space.low
        high = self.action_space.high

        scale_factor = (high - low) / 2
        reloc_factor = high - scale_factor

        action = action * scale_factor + reloc_factor
        action = np.clip(action, low, high)

        return action

    def reverse_action(self, action: np.ndarray) -> np.ndarray:
        """Change the range (low, high) to (-1, 1)."""
        low = self.action_space.low
        high = self.action_space.high

        scale_factor = (high - low) / 2
        reloc_factor = high - scale_factor

        action = (action - reloc_factor) / scale_factor
        action = np.clip(action, -1.0, 1.0)

        return action

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb-run-name", type=str, default="cheetah-sac")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--discount-factor", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=5e-3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--initial-random-steps", type=int, default=1000)
    parser.add_argument("--memory-size", type=int, default=1000000)
    parser.add_argument("--num-steps", type=int, default=1000000)
    parser.add_argument("--policy-update-freq", type=int, default=2)
    parser.add_argument("--seed", type=int, default=77)
    args = parser.parse_args()
    
    local_time = time.localtime()
    timestamp = time.strftime("%m%d%Y_%H%M%S", local_time)
    
    # W&B init
    wandb.init(
        project="RL-HW3-SAC-Cheetah",
        config=vars(args),
        name=f"{args.wandb_run_name}-{timestamp}",
        save_code=True,
    )
    for k, v in wandb.config.items():
        setattr(args, k, v)
    
    print("Training arguments:")
    for key, value in vars(args).items():
        print(f"{key}: {value}")

    # environment
    env = gym.make("HalfCheetah-v5", render_mode="rgb_array")
    env = ActionNormalizer(env)
    agent = SACAgent(env, args)
    agent.train()