import os
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

class NetworkBase(nn.Module):
    def __init__(self, input_dims, layer_dims, output_dims, name, checkpoint_dir, learning_rate=10e-3):
        super(NetworkBase, self).__init__()

        layers = []
        previous_dim = input_dims
        for dim in layer_dims:
            layers.append(nn.Linear(previous_dim, dim))
            previous_dim = dim


        self.input_dims = input_dims
        self.layer_dims = layer_dims
        self.output_dims = output_dims
        self.name = name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_td3')
        
        layers = [nn.Linear(input_dims, layer_dims[0])]
        layers += [nn.Linear(layer_dims[i], layer_dims[i+1]) for i in range(len(layer_dims)-1)]
        self.layers = nn.ModuleList(layers)
        self.output = nn.Linear(layer_dims[-1], output_dims)

        self.optimizer = optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=0.005)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        return self.output(x)

    def save_checkpoint(self, episode_num):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        T.save(self.state_dict(), self.checkpoint_file)
        print(f'{self.name} Checkpoint saved to {self.checkpoint_file}')

    def load_checkpoint(self):
        try:
            self.load_state_dict(T.load(self.checkpoint_file))
            print(f'{self.name} Successfully loaded checkpoint from:', self.checkpoint_file)
        except Exception as e:
            print(f'{self.name} Failed to load checkpoint from:', self.checkpoint_file)
            print('Error:', e)

class CriticNetwork(NetworkBase):
    def __init__(self, input_dims, n_actions, layer_dims, checkpoint_dir, name='critic', learning_rate=10e-3):
        super().__init__(input_dims + n_actions, layer_dims, 1, name, checkpoint_dir, learning_rate)

    def forward(self, state, action):
        x = T.cat([state, action], dim=1)
        return super().forward(x)

class ActorNetwork(NetworkBase):
    def __init__(self, input_dims, n_actions, layer_dims, checkpoint_dir, name='actor', learning_rate=10e-3):
        super().__init__(input_dims, layer_dims, n_actions, name, checkpoint_dir, learning_rate)

    def forward(self, state):
        x = super().forward(state)
        return T.tanh(x)
