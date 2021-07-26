import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from gym import Wrapper
from stable_baselines3 import SAC
from tqdm.auto import tqdm

class Buffer():
    """Create a replay buffer for the discriminator to learn
    from past observations conditionned by skills"""
    
    def __init__(self, *shape):
        """
        Args:
            shape (tuple of int) : the expected shape is the following : (buffer_size, state_size + 1)
                buffer_size has to be greater than total_timesteps.
        """
        self.buffer = torch.zeros(shape)
        self._current_size = 0
    
    def sample(self, batch_size):
        """
        Args:
            batch_size (int) : number of observations to sample and return
        Returns:
            Tensor of shape (batch_size, state_size + 1)
        """
        assert batch_size <= self._current_size, "batch size is greater than the buffer length"
        index = np.random.randint(0, self._current_size, batch_size)
        return self.buffer[index]
    
    def size(self):
        """Return the current_size of the buffer, i.e. the non-zero lines"""
        return self._current_size
    
    def add_sample(self, state, label):
        """Add sample to buffer
        Args:
            state (NumPy array) : expected shape : (state_size,)
            label (int) : the current skill bound to current state
        """
        index = self._current_size
        self.buffer[index, :-1] = torch.from_numpy(state)
        self.buffer[index, -1] = label
        self._current_size += 1
    
    def __repr__(self):
        return str(self.buffer)

class DIAYN(Wrapper):
    """Create a Wrapper for Gym Environment
    based on Diversity Is All You Need
    """
    def __init__(self, env, n_skills, total_timesteps=None, batch_size=64, hidden_dim=128, lr=1e-3):
        """
        Args:
            env (gym env)
            n_skills (int) : number of skills
            total_timesteps (int) : same parameter as algorithm.
                If not None, DIAYN is in "training" mode : a progress bar will
                appear during training. If None, there would be no progress bar.
            hidden_dim (int) : dimension of latent space
            lr (float) : learning rate
        """
        Wrapper.__init__(self, env)
        self.n_skills = n_skills
        self.hidden_dim = hidden_dim
        self.state_size = env.observation_space.shape[0]
        self.lr = lr
        self.batch_size = batch_size
        self.probability_of_skill = 1 / self.n_skills
        
        self.discriminator = nn.Sequential(
                                nn.Linear(self.state_size, self.hidden_dim),
                                nn.ReLU(),
                                nn.Linear(self.hidden_dim, self.hidden_dim),
                                nn.ReLU(),
                                nn.Linear(self.hidden_dim, self.n_skills)
        )
        
        
        self.discriminator_optimizer = optim.Adam(self.discriminator.parameters(), lr=self.lr)
        self.discriminator_optimizer.zero_grad()
        

        # Set up the environment
        self.env.observation_space.shape = (self.state_size + self.n_skills,)

        # Init skill and "loggers"
        self.skill = np.random.randint(self.n_skills)

        self.training_mode = total_timesteps is not None
        self.pbar = tqdm(total=total_timesteps, disable = not self.training_mode)
        self.pbar.set_postfix_str("Ready to train !")
        self.current_experiment_number= [0]
        self.discriminator_losses = []

        self.current_step = 0
    
        if self.training_mode:
            self.buffer = Buffer(total_timesteps, self.state_size + 1) # state + current skill

    def augment_state(self, state):
        """
        From the author : "To pass skill $z$ to the Q function, value function,
        and policy, we simply concatenate $z$ to the current state $s_t$"
        
        Args:
            state (gym state : NumPy array)
        
        Returns:
            An augmented state with:
                - current state of the environment
                - the current skill
        """

        skill_one_hot = np.zeros(self.n_skills)
        skill_one_hot[self.skill] = 1

        return np.concatenate((state, skill_one_hot))

    def compute_discriminator_scores(self, state, skill):
        """Compute discriminator scores according to current state
        Args:
            state (gym state)
            skill (int)
        Returns:
            (probability of current skill, scores)
        """
        scores = self.discriminator(torch.tensor(state).unsqueeze(0).float())
        probability_of_current_skill = F.softmax(scores, dim=-1)[:, skill]

        return probability_of_current_skill.item(), scores

    def compute_reward(self, state, skill):
        """Compute reward using discriminator outputs
        
        Args:
            state (gym state)
            skill (int)
        Returns:
            reward, scores
        """
        probability_of_current_skill, scores =  self.compute_discriminator_scores(state, skill)
        reward = np.log(max(probability_of_current_skill, 1E-6)) - np.log(self.probability_of_skill)

        return reward, scores 
    
    def step(self, action):
        """Perform a step in environment and compute skill reward
        
        Args:
            action (gym action)
        Return:
            Results from action : state, reward, done, info
        """
        self.current_step += 1
        # Step in environment, skip reward and augment observation/state
        state, _, done, info = self.env.step(action)
        augmented_state = self.augment_state(state)
        
        # Compute skill reward : r_t = \log(q_\phi(z|s_{t+1})) - \log(p(z))
        reward, scores = self.compute_reward(state, self.skill)
        
        # Update discriminator
        if self.training_mode:
            # Add current state and skill to buffer
            self.buffer.add_sample(state, self.skill)
        
            if self.buffer.size() >= self.batch_size:
                batch = self.buffer.sample(self.batch_size)
                batch_states = batch[:, :-1]
                batch_labels = batch[:, -1]

                scores = self.discriminator(batch_states)
                loss = F.cross_entropy(scores, batch_labels.long())
                loss.backward()
                self.discriminator_optimizer.step()
                self.discriminator_optimizer.zero_grad()

                self.discriminator_losses.append(loss.data)
                self.pbar.set_postfix_str(f"Experiment : {self.current_experiment_number[-1]}, skill : {self.skill}, discriminator Loss : {loss.data}")

            # Perform various logs
            self.pbar.update(1)
            current_n_exp = self.current_experiment_number[-1]
            if done:
                self.current_experiment_number.append(current_n_exp + 1)
            else:
                self.current_experiment_number.append(current_n_exp)

        return augmented_state, reward, done, info

    def reset(self, **kwargs):
        """Reset gym environment and sample a new skill"""
        state = self.env.reset(**kwargs)
        if self.training_mode:
            self.skill = np.random.randint(self.n_skills)

        return self.augment_state(state)
