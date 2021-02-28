import numpy as np
import torch as T
from deep_q_network import DeepQNetwork
from reply_memory import ReplayBuffer

class DQNAgent():
    def __init__(self, gamma, epsilon, lr, n_actions, input_dims, 
                mem_size, batch_size, eps_min=0.01, eps_dec = 5e-7,
                replace=1000, algo=None, env_name=None, chkpt_dir='checkpoints'):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.mem_size = mem_size
        self.batch_size = batch_size
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.replace_target_cnt = replace
        self.algo = algo
        self.env_name = env_name
        self.chkpt_dir = chkpt_dir
        self.action_space = [i for i in range(self.n_actions)]
        self.learn_step_counter = 0

        self.memory = ReplayBuffer(self.mem_size, self.input_dims, self.n_actions)
        self.q_eval = DeepQNetwork(self.lr, self.n_actions, name=self.env_name+'_'+self.algo+'_q_eval', input_dims=self.input_dims, chkpt_dir=self.chkpt_dir)
        self.q_next = DeepQNetwork(self.lr, self.n_actions, name=self.env_name+'_'+self.algo+'_q_next', input_dims=self.input_dims, chkpt_dir=self.chkpt_dir)
        print(f"RUNNING ON {str(self.q_eval.device).upper()}")

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = T.tensor([observation], dtype=T.float).to(self.q_eval.device)
            actions = self.q_eval.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def sample_memory(self):
        states, actions, rewards, states_, dones = self.memory.sample_buffer(self.batch_size)
        states = T.tensor(states).to(self.q_eval.device)
        actions = T.tensor(actions).to(self.q_eval.device)
        rewards = T.tensor(rewards).to(self.q_eval.device)
        states_ = T.tensor(states_).to(self.q_eval.device)
        dones = T.tensor(dones, dtype=T.bool).to(self.q_eval.device)

        return states, actions, rewards, states_, dones

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

    def save_models(self):
        self.q_eval.save_checkpoint()
        self.q_next.save_checkpoint()

    def load_models(self):
        self.q_eval.load_checkpoint()
        self.q_next.load_checkpoint()

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        self.q_eval.optimizer.zero_grad()
        self.replace_target_network()

        states, actions, rewards, states_, dones = self.sample_memory()
        #print(states)
        #print(actions)
        #print(rewards)
        #print(states_)
        #print(dones)
        #print("-----------------------------------------------------")
        indices = np.arange(self.batch_size)
        q_pred = self.q_eval.forward(states)[indices, actions]
        q_next = self.q_next.forward(states_).max(dim=1)[0]
    
        q_next[dones] = 0.0
    
        q_target = rewards + self.gamma * q_next
        
        loss = self.q_eval.loss(q_pred, q_target).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1
        self.decrement_epsilon()



    

    

