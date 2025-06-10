# td_learning.py
import numpy as np
from collections import defaultdict
from maze import Maze, Actions
import random

class OptimalPolicy:
    """Optimal policy for the maze environment"""
    
    def __init__(self):
        # Define optimal actions for each state
        # This policy leads from any reachable state to the +40 terminal state
        self.policy_map = {
            # Path from start to +40 terminal
            (3, 2): Actions.UP,    # Start: go up
            (2, 2): Actions.LEFT,  # Go left to avoid penalty
            (2, 1): Actions.UP,    # Go up
            (1, 1): Actions.UP,    # Go up
            (0, 1): Actions.RIGHT, # Go right
            (0, 2): Actions.RIGHT, # Go right to reach goal
            
            # Additional states for completeness
            (3, 1): Actions.RIGHT, # Move toward start path
            (3, 3): Actions.LEFT,  # Move toward start path
            (2, 3): Actions.LEFT,  # Avoid penalty, go left
            (2, 0): Actions.RIGHT, # Move toward main path
            (1, 0): Actions.RIGHT, # Move toward main path
            (0, 0): Actions.RIGHT, # Move toward goal
            
            # Terminal states have no actions
            (0, 3): None,  # +40 terminal
            (3, 0): None,  # +10 terminal
        }
    
    def get_action(self, state):
        """Get action for given state"""
        return self.policy_map.get(state, Actions.UP)  # Default action if not in map

class TDLearning:
    """Temporal Difference Learning for Policy Evaluation"""
    
    def __init__(self, maze, policy, alpha=0.1, gamma=1.0):
        self.maze = maze
        self.policy = policy
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        
        # Initialize value function
        self.V = defaultdict(float)
        # Terminal states have value 0
        for terminal in maze.terminal_states:
            self.V[terminal] = 0.0
            
        # Statistics
        self.episode_count = 0
        self.learning_curve = []
        self.value_history = defaultdict(list)  # Track value evolution
        self.visit_count = defaultdict(int)     # Track state visits
        
    def generate_episode_step(self, state):
        """Generate one step of an episode"""
        if self.maze.is_terminal(state):
            return None, None, None, True
            
        action = self.policy.get_action(state)
        next_state, reward, done = self.maze.step(state, action)
        
        return action, next_state, reward, done
    
    def td_update(self, state, reward, next_state):
        """Perform TD(0) update"""
        if not self.maze.is_terminal(state):
            # TD(0) update: V(S) ← V(S) + α[R + γV(S') - V(S)]
            td_target = reward + self.gamma * self.V[next_state]
            td_error = td_target - self.V[state]
            self.V[state] += self.alpha * td_error
            
            # Track visit count
            self.visit_count[state] += 1
            
            return abs(td_error)
        return 0
    
    def run_episode(self, start_state=None, max_steps=100):
        """Run one episode and update values"""
        if start_state is None:
            current_state = self.maze.start_position
        else:
            current_state = start_state
            
        total_td_error = 0
        steps = 0
        episode_states = []
        
        while not self.maze.is_terminal(current_state) and steps < max_steps:
            episode_states.append(current_state)
            action, next_state, reward, done = self.generate_episode_step(current_state)
            
            if action is None:  # Terminal state reached
                break
                
            # Perform TD update
            td_error = self.td_update(current_state, reward, next_state)
            total_td_error += td_error
            
            current_state = next_state
            steps += 1
        
        # Record value history for visited states
        for state in episode_states:
            self.value_history[state].append(self.V[state])
            
        return total_td_error, steps
    
    def train(self, num_episodes=5000, verbose=True, 
              exploration_episodes=1000, exploration_prob=0.3):
        """Train the value function using TD learning
        
        Args:
            num_episodes: Total number of episodes
            verbose: Print progress
            exploration_episodes: Number of episodes to use random starts
            exploration_prob: Probability of random start during exploration
        """
        if verbose:
            print(f"Training TD Learning with α={self.alpha}, γ={self.gamma}")
            print(f"Using exploration for first {exploration_episodes} episodes")
            
        for episode in range(num_episodes):
            # Use random starting states for better exploration
            if episode < exploration_episodes and random.random() < exploration_prob:
                # Choose random non-terminal state
                all_states = [s for s in self.maze.get_all_states() 
                             if not self.maze.is_terminal(s)]
                start_state = random.choice(all_states)
            else:
                start_state = None  # Use default start
                
            total_error, steps = self.run_episode(start_state)
            self.learning_curve.append(total_error)
            
            if verbose and (episode + 1) % 1000 == 0:
                avg_error = np.mean(self.learning_curve[-100:])
                visited_states = len([s for s in self.V.keys() 
                                    if s not in self.maze.terminal_states])
                print(f"Episode {episode + 1}: Avg TD Error: {avg_error:.4f}, "
                      f"States learned: {visited_states}")
                
        self.episode_count = num_episodes
        
        if verbose:
            print("Training completed!")
            
    def get_value_function(self):
        """Get the learned value function"""
        return dict(self.V)
    
    def get_statistics(self):
        """Get training statistics"""
        visited_states = len([s for s in self.V.keys() if s not in self.maze.terminal_states])
        total_states = len(self.maze.get_all_states()) - len(self.maze.terminal_states)
        
        # Calculate average visits per state
        avg_visits = np.mean(list(self.visit_count.values())) if self.visit_count else 0
        max_visits = max(self.visit_count.values()) if self.visit_count else 0
        min_visits = min([v for v in self.visit_count.values() if v > 0]) if self.visit_count else 0
        
        return {
            'episodes_trained': self.episode_count,
            'states_visited': visited_states,
            'total_states': total_states,
            'coverage': visited_states / total_states if total_states > 0 else 0,
            'learning_curve': self.learning_curve.copy(),
            'final_avg_error': np.mean(self.learning_curve[-100:]) if len(self.learning_curve) >= 100 else 0,
            'value_history': dict(self.value_history),
            'visit_count': dict(self.visit_count),
            'avg_visits': avg_visits,
            'max_visits': max_visits,
            'min_visits': min_visits
        }
    
    def evaluate_from_all_states(self, num_episodes_per_state=100):
        """Evaluate policy starting from all possible states (for complete value function)"""
        print("Evaluating from all possible starting states...")
        
        all_states = [s for s in self.maze.get_all_states() if not self.maze.is_terminal(s)]
        
        for state in all_states:
            for _ in range(num_episodes_per_state):
                self.run_episode(start_state=state)
                
        print("Complete evaluation finished!")

def compare_td_methods():
    """Compare TD learning with different gamma values"""
    maze = Maze()
    policy = OptimalPolicy()
    
    print("="*60)
    print("TEMPORAL DIFFERENCE LEARNING COMPARISON")
    print("="*60)
    
    # TD with γ = 1.0
    print("\n1. TD Learning with γ = 1.0 (No Discounting)")
    print("-" * 50)
    td_gamma_1 = TDLearning(maze, policy, alpha=0.1, gamma=1.0)
    td_gamma_1.train(num_episodes=5000, exploration_episodes=2000, exploration_prob=0.5)
    values_1 = td_gamma_1.get_value_function()
    stats_1 = td_gamma_1.get_statistics()
    
    print(f"\nState Coverage: {stats_1['coverage']:.1%}")
    print(f"Average visits per state: {stats_1['avg_visits']:.1f}")
    print(f"Visit range: {stats_1['min_visits']} to {stats_1['max_visits']}")
    
    # TD with γ = 0.5
    print("\n2. TD Learning with γ = 0.5 (Heavy Discounting)")
    print("-" * 50)
    td_gamma_05 = TDLearning(maze, policy, alpha=0.1, gamma=0.5)
    td_gamma_05.train(num_episodes=5000, exploration_episodes=2000, exploration_prob=0.5)
    values_05 = td_gamma_05.get_value_function()
    stats_05 = td_gamma_05.get_statistics()
    
    print(f"\nState Coverage: {stats_05['coverage']:.1%}")
    print(f"Average visits per state: {stats_05['avg_visits']:.1f}")
    print(f"Visit range: {stats_05['min_visits']} to {stats_05['max_visits']}")
    
    # Compare learned values
    print("\n3. Value Function Comparison")
    print("-" * 50)
    print("State    | V(s) γ=1.0 | V(s) γ=0.5 | Difference")
    print("-" * 50)
    
    # Show values for key states
    key_states = [(3, 2), (2, 2), (2, 1), (1, 1), (0, 1), (0, 2)]  # Path to goal
    for state in key_states:
        v1 = values_1.get(state, 0)
        v05 = values_05.get(state, 0)
        diff = v1 - v05
        print(f"{state}   | {v1:11.2f} | {v05:11.2f} | {diff:10.2f}")
    
    return {
        'gamma_1.0': {'values': values_1, 'stats': stats_1, 'td_learner': td_gamma_1},
        'gamma_0.5': {'values': values_05, 'stats': stats_05, 'td_learner': td_gamma_05}
    }

if __name__ == "__main__":
    results = compare_td_methods()