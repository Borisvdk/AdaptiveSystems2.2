# maze.py
import numpy as np
from enum import Enum
import random

class Actions(Enum):
    LEFT = 0
    UP = 1
    RIGHT = 2
    DOWN = 3

class Maze:
    def __init__(self, stochastic=False, noise_prob=0.2):
        # Grid layout with rewards
        self.rewards_grid = np.array([
            [-1, -1, -1, 40],   # Row 0
            [-1, -1, -10, -10], # Row 1
            [-1, -1, -1, -1],   # Row 2
            [10, -2, -1, -1]    # Row 3
        ])
        
        self.height = 4
        self.width = 4
        
        # Terminal states
        self.terminal_states = [(0, 3), (3, 0)]
        
        # Start position
        self.start_position = (3, 2)
        
        # Stochastic environment parameters
        self.stochastic = stochastic
        self.noise_prob = noise_prob
        
        # All possible actions
        self.actions = list(Actions)
        
    def reset(self):
        """Reset environment to start position"""
        return self.start_position
        
    def is_terminal(self, state):
        """Check if a state is terminal"""
        return state in self.terminal_states
    
    def get_reward(self, state):
        """Get reward for a state"""
        row, col = state
        return self.rewards_grid[row, col]
    
    def get_valid_actions(self, state):
        """Get all valid actions from a state"""
        return self.actions.copy()
    
    def get_next_state(self, state, action):
        """Get next state given current state and action (deterministic)"""
        row, col = state
        
        if action == Actions.UP:
            row = max(0, row - 1)
        elif action == Actions.DOWN:
            row = min(self.height - 1, row + 1)
        elif action == Actions.LEFT:
            col = max(0, col - 1)
        elif action == Actions.RIGHT:
            col = min(self.width - 1, col + 1)
            
        return (row, col)
    
    def step(self, state, action):
        """Take a step in the environment"""
        if self.stochastic:
            # With probability noise_prob, take a random action instead
            if random.random() < self.noise_prob:
                action = random.choice(self.actions)
        
        next_state = self.get_next_state(state, action)
        reward = self.get_reward(next_state)
        done = self.is_terminal(next_state)
        
        return next_state, reward, done
    
    def get_all_states(self):
        """Get all possible states in the environment"""
        states = []
        for i in range(self.height):
            for j in range(self.width):
                states.append((i, j))
        return states
    
    def get_all_state_action_pairs(self):
        """Get all possible state-action pairs"""
        pairs = []
        for state in self.get_all_states():
            if not self.is_terminal(state):
                for action in self.actions:
                    pairs.append((state, action))
        return pairs