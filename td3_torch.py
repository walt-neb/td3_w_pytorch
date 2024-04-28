#filename: td3_torch.py

import torch as T
import torch.nn.functional as F
import numpy as np
from buffer import ReplayBuffer
from networks import ActorNetwork, CriticNetwork

def save_network_statistics(network, network_name, filename):
    stats = {}
    for name, param in network.named_parameters():
        param = param.data.cpu().numpy()
        stats[f"{network_name}_{name}"] = {
            'mean': param.mean(),
            'std': param.std(),
            'max': param.max(),
            'min': param.min()
        }

    with open(filename, 'w') as file:
        for stat_name, values in stats.items():
            file.write(f"{stat_name} - Mean: {values['mean']}, Std: {values['std']}, "
                       f"Max: {values['max']}, Min: {values['min']}\n")
    print(f'Checkpoint stats saved for {network_name}')


def save_checkpoint_statistics(agent, filename):
    save_network_statistics(agent.actor, "Actor", filename+'_actor')
    save_network_statistics(agent.critic_1, "Critic1", filename+'_critic1')
    save_network_statistics(agent.critic_2, "Critic2", filename+'_critic2')


class OUNoise:
    def __init__(self, mu, theta, sigma):
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.state = np.ones_like(self.mu) * self.mu

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(*self.mu.shape)
        self.state = x + dx
        return self.state

class Agent_TD3:
    def __init__(self, actor_learning_rate,
                 critic_learning_rate,
                 input_dims,
                 tau, env,
                 gamma=0.99,
                 update_actor_interval=2,
                 warmup=1000,
                 n_actions=2,
                 max_replay_buffer_size=100000,
                 layer_dims = [256, 256, 256],
                 batch_size=100,
                 noise=0.1,
                 checkpoint_dir='./checkpoints/'):

        self.noise = OUNoise(mu=np.zeros(n_actions), theta=0.15, sigma=0.2)
        self.gamma = gamma
        self.tau = tau
        self.max_action = env.action_space.high
        self.min_action = env.action_space.low
        self.replay_buffer = ReplayBuffer(max_replay_buffer_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.learn_step_cntr = 0
        self.time_step = 0
        self.total_time_steps = 0
        self.warmup = warmup
        self.n_actions = n_actions
        input_dims = input_dims[0]
        layer_dims = layer_dims
        self.update_actor_interval = update_actor_interval

        # Usage example:
        #layer_dims = [256, 256, 256]  # You can change this list to add/remove layers
        #critic = CriticNetwork(input_dims=24, n_actions=2, layer_dims=layer_dims, checkpoint_dir='checkpoints')
        #actor = ActorNetwork(input_dims=24, n_actions=2, layer_dims=layer_dims, checkpoint_dir='checkpoints')

        print("Layer dimensions:", layer_dims)
        print("Type of each element:", [type(dim) for dim in layer_dims])


        # Create the networks
        self.actor = ActorNetwork(input_dims=input_dims,
                                  n_actions=n_actions,
                                  layer_dims=layer_dims,
                                  checkpoint_dir='checkpoints',
                                  learning_rate=actor_learning_rate)

        self.critic_1 = CriticNetwork(input_dims=input_dims,
                                      n_actions=n_actions,
                                      layer_dims=layer_dims,
                                      name='critic_1',
                                      checkpoint_dir='checkpoints',
                                      learning_rate=critic_learning_rate)

        self.critic_2 = CriticNetwork(input_dims=input_dims,
                                      n_actions=n_actions,
                                      layer_dims=layer_dims,
                                      name='critic_2',
                                      checkpoint_dir='checkpoints',
                                      learning_rate=critic_learning_rate)

        # Create the target network
        self.target_actor = ActorNetwork(input_dims=input_dims,
                                         n_actions=n_actions,
                                         layer_dims=layer_dims,
                                         name='target_actor',
                                         checkpoint_dir='checkpoints',
                                         learning_rate=actor_learning_rate)

        self.target_critic_1 = CriticNetwork(input_dims=input_dims,
                                             n_actions=n_actions,
                                             layer_dims=layer_dims,
                                             name='target_critic_1',
                                             checkpoint_dir='checkpoints',
                                             learning_rate=critic_learning_rate)

        self.target_critic_2 = CriticNetwork(input_dims=input_dims,
                                             n_actions=n_actions,
                                             layer_dims=layer_dims,
                                             name='target_critic_2',
                                             checkpoint_dir='checkpoints',
                                             learning_rate=critic_learning_rate)

        self.update_network_parameters(tau=1)

    def choose_action(self, observation, validation=False):
        # Use the total number of steps experienced by the agent, not just in the current session
        if self.total_time_steps < self.warmup and not validation:
            action = self.noise.sample()
        else:
            state = T.tensor(observation, dtype=T.float).to(self.actor.device)
            action = self.actor.forward(state).to(self.actor.device)
            action = action.cpu().detach().numpy()
            action += self.noise.sample() * self.max_action  # Add exploration noise

        # Ensure actions are within bounds
        action = np.clip(action, self.min_action, self.max_action)

        self.total_time_steps += 1  # Increment the total time steps
        return action


    def remmember(self, state, action, reward, next_state, done):
        self.replay_buffer.store_transition(state, action, reward, next_state, done)

    def learn(self):
        if self.replay_buffer.mem_ctr < self.batch_size * 10:
            return

        state, action, reward, next_state, done = self.replay_buffer.sample_buffer(self.batch_size)
        reward = T.tensor(reward, dtype=T.float).to(self.critic_1.device)
        done = T.tensor(done).to(self.critic_1.device)
        next_state = T.tensor(next_state, dtype=T.float).to(self.critic_1.device)
        state = T.tensor(state, dtype=T.float).to(self.critic_1.device)
        action = T.tensor(action, dtype=T.float).to(self.critic_1.device)

        target_actions = self.target_actor.forward(next_state)

        target_actions = target_actions + T.clamp(T.tensor(np.random.normal(scale=0.2)), -0.5, 0.5)
        target_actions = T.clamp(target_actions, self.min_action[0], self.max_action[0])

        next_q1 = self.target_critic_1.forward(next_state, target_actions)
        next_q2 = self.target_critic_2.forward(next_state, target_actions)

        q1 = self.critic_1.forward(state, action)
        q2 = self.critic_2.forward(state, action)

        next_q1[done]= 0.0
        next_q2[done]= 0.0

        next_q1 = next_q1.view(-1)
        next_q2 = next_q2.view(-1)

        next_critic_value = T.min(next_q1, next_q2)
        target = reward + self.gamma * next_critic_value
        target = target.view(self.batch_size, 1)

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()

        q1_loss = F.mse_loss(target, q1)
        q2_loss = F.mse_loss(target, q2)

        critic_loss = q1_loss + q2_loss
        critic_loss.backward()

        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        self.learn_step_cntr += 1

        if self.learn_step_cntr % self.update_actor_interval == 0:
            # Actor update skipped this iteration
            return {
                'critic_loss': critic_loss.item(),
                'actor_loss': None,  # No update performed
                'mean_reward': reward.mean().item(),
            }

        self.actor.optimizer.zero_grad()
        actor_q1_loss = self.critic_1.forward(state, self.actor.forward(state))
        actor_loss = -T.mean(actor_q1_loss)
        actor_loss.backward()
        self.actor.optimizer.step()
        self.update_network_parameters()

        # After the updates, we collect the metrics to return
        metrics = {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'mean_reward': reward.mean().item(),
            'mean_q1': q1.mean().item(),
            'mean_q2': q2.mean().item(),
            'mean_next_q1': next_q1.mean().item(),
            'mean_next_q2': next_q2.mean().item(),
        }

        return metrics

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        # Update target critic 1
        with T.no_grad():
            for target_param, param in zip(self.target_critic_1.parameters(), self.critic_1.parameters()):
                target_param.data.mul_(1.0 - tau)
                target_param.data.add_(tau * param.data)

        # Update target critic 2
        with T.no_grad():
            for target_param, param in zip(self.target_critic_2.parameters(), self.critic_2.parameters()):
                target_param.data.mul_(1.0 - tau)
                target_param.data.add_(tau * param.data)

        # Update target actor
        with T.no_grad():
            for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
                target_param.data.mul_(1.0 - tau)
                target_param.data.add_(tau * param.data)


    def save_checkpoint(self, filename, episode_num):
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_1_state_dict': self.critic_1.state_dict(),
            'critic_2_state_dict': self.critic_2.state_dict(),
            'target_actor_state_dict': self.target_actor.state_dict(),
            'target_critic_1_state_dict': self.target_critic_1.state_dict(),
            'target_critic_2_state_dict': self.target_critic_2.state_dict(),
            'actor_optimizer_state': self.actor.optimizer.state_dict(),
            'critic1_optimizer_state': self.critic_1.optimizer.state_dict(),
            'critic2_optimizer_state': self.critic_2.optimizer.state_dict(), # Assuming same optimizer for both critics
            'numpy_rng_state': np.random.get_state(),  # Save NumPy's RNG state
            'torch_rng_state': T.get_rng_state(),  # using PyTorch's random number generator
            'episode_num': episode_num,  # Save the current episode number
            'total_time_steps': self.total_time_steps,
        }
        T.save(checkpoint, filename)
        print(f'Checkpoint saved to {filename}')
        stats_file = f'{filename}.stats'
        save_checkpoint_statistics(self, stats_file)


    def load_checkpoint(self, filename):
        try:
            checkpoint = T.load(filename)
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.critic_1.load_state_dict(checkpoint['critic_1_state_dict'])
            self.critic_2.load_state_dict(checkpoint['critic_2_state_dict'])
            self.target_actor.load_state_dict(checkpoint['target_actor_state_dict'])
            self.target_critic_1.load_state_dict(checkpoint['target_critic_1_state_dict'])
            self.target_critic_2.load_state_dict(checkpoint['target_critic_2_state_dict'])
            self.actor.optimizer.load_state_dict(checkpoint['actor_optimizer_state'])
            self.critic_1.optimizer.load_state_dict(checkpoint['critic1_optimizer_state'])
            self.critic_2.optimizer.load_state_dict(checkpoint['critic2_optimizer_state'])
            np.random.set_state(checkpoint['numpy_rng_state'])  # Restore NumPy's RNG state
            T.set_rng_state(checkpoint['torch_rng_state'])
            episode_num = checkpoint['episode_num']  # Retrieve the episode number
            self.total_time_steps = checkpoint.get('total_time_steps', 0)  # Restore total time steps

            # Update target networks
            self.update_network_parameters(tau=1)  # Immediately copy the weights to target networks

            print(f'Checkpoint loaded from {filename}')
            return episode_num

        except Exception as e:
            print(f'{e} \nFailed to load checkpoint from:', filename)


