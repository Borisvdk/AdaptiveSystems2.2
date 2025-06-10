# double_q_learning.py
import numpy as np
import random
from collections import defaultdict
from maze import Maze, Actions

class DoubleQLearning:
    """Double Q-Learning: Reduces bias by using two Q-tables"""
    
    def __init__(self, maze, alpha=0.1, gamma=1.0, epsilon=0.1, epsilon_decay=0.995, epsilon_min=0.01):
        self.maze = maze
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Initialize two Q-functions
        self.Q1 = defaultdict(lambda: defaultdict(float))
        self.Q2 = defaultdict(lambda: defaultdict(float))
        
        # Initialize Q(terminal, ·) = 0 for all actions in both tables
        for terminal_state in maze.terminal_states:
            for action in Actions:
                self.Q1[terminal_state][action] = 0.0
                self.Q2[terminal_state][action] = 0.0
        
        # Statistics
        self.episode_count = 0
        self.learning_curve = []
        self.epsilon_curve = []
        self.episode_rewards = []
        self.episode_lengths = []
        
    def get_combined_q_value(self, state, action):
        """Get combined Q-value from both tables"""
        return self.Q1[state][action] + self.Q2[state][action]
    
    def get_epsilon_greedy_action(self, state):
        """Choose action using ε-greedy policy based on combined Q-values"""
        if self.maze.is_terminal(state):
            return None
            
        # ε-greedy action selection
        if random.random() < self.epsilon:
            # Explore: choose random action
            return random.choice(list(Actions))
        else:
            # Exploit: choose best action based on combined Q-values
            return self.get_greedy_action(state)
    
    def get_greedy_action(self, state):
        """Get greedy action based on combined Q-values"""
        if self.maze.is_terminal(state):
            return None
            
        # Find action with highest combined Q-value
        best_actions = []
        best_value = float('-inf')
        
        for action in Actions:
            q_value = self.get_combined_q_value(state, action)
            if q_value > best_value:
                best_value = q_value
                best_actions = [action]
            elif q_value == best_value:
                best_actions.append(action)
        
        # Break ties randomly
        return random.choice(best_actions)
    
    def get_best_action_q1(self, state):
        """Get best action according to Q1"""
        if self.maze.is_terminal(state):
            return None
            
        best_actions = []
        best_value = float('-inf')
        
        for action in Actions:
            q_value = self.Q1[state][action]
            if q_value > best_value:
                best_value = q_value
                best_actions = [action]
            elif q_value == best_value:
                best_actions.append(action)
        
        return random.choice(best_actions)
    
    def get_best_action_q2(self, state):
        """Get best action according to Q2"""
        if self.maze.is_terminal(state):
            return None
            
        best_actions = []
        best_value = float('-inf')
        
        for action in Actions:
            q_value = self.Q2[state][action]
            if q_value > best_value:
                best_value = q_value
                best_actions = [action]
            elif q_value == best_value:
                best_actions.append(action)
        
        return random.choice(best_actions)
    
    def double_q_learning_update(self, state, action, reward, next_state):
        """Perform Double Q-learning update"""
        if self.maze.is_terminal(state):
            return 0
            
        # With 0.5 probability, update Q1 using Q2 for next state value
        if random.random() < 0.5:
            # Update Q1: Q1(S,A) ← Q1(S,A) + α[R + γQ2(S', argmax_a Q1(S',a)) - Q1(S,A)]
            if self.maze.is_terminal(next_state):
                q_next = 0.0
            else:
                best_action_q1 = self.get_best_action_q1(next_state)
                q_next = self.Q2[next_state][best_action_q1] if best_action_q1 else 0.0
            
            q_target = reward + self.gamma * q_next
            q_error = q_target - self.Q1[state][action]
            self.Q1[state][action] += self.alpha * q_error
        else:
            # Update Q2: Q2(S,A) ← Q2(S,A) + α[R + γQ1(S', argmax_a Q2(S',a)) - Q2(S,A)]
            if self.maze.is_terminal(next_state):
                q_next = 0.0
            else:
                best_action_q2 = self.get_best_action_q2(next_state)
                q_next = self.Q1[next_state][best_action_q2] if best_action_q2 else 0.0
            
            q_target = reward + self.gamma * q_next
            q_error = q_target - self.Q2[state][action]
            self.Q2[state][action] += self.alpha * q_error
        
        return abs(q_error)
    
    def run_episode(self, max_steps=200):
        """Run one episode using Double Q-learning"""
        state = self.maze.reset()
        
        total_reward = 0
        total_error = 0
        steps = 0
        
        while not self.maze.is_terminal(state) and steps < max_steps:
            # Choose action using ε-greedy policy based on combined Q-values
            action = self.get_epsilon_greedy_action(state)
            
            if action is None:  # Terminal state
                break
                
            # Take action, observe reward and next state
            next_state, reward, done = self.maze.step(state, action)
            total_reward += reward
            
            # Double Q-learning update
            error = self.double_q_learning_update(state, action, reward, next_state)
            total_error += error
            
            # Move to next state
            state = next_state
            steps += 1
            
        return total_reward, total_error, steps
    
    def train(self, num_episodes=10000, verbose=True):
        """Train using Double Q-learning algorithm"""
        if verbose:
            print(f"Training Double Q-Learning with α={self.alpha}, γ={self.gamma}, ε={self.epsilon}")
            
        for episode in range(num_episodes):
            # Run episode
            reward, error, steps = self.run_episode()
            
            # Store statistics
            self.episode_rewards.append(reward)
            self.learning_curve.append(error)
            self.episode_lengths.append(steps)
            self.epsilon_curve.append(self.epsilon)
            
            # Decay epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            
            # Verbose output
            if verbose and (episode + 1) % 1000 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                avg_steps = np.mean(self.episode_lengths[-100:])
                print(f"Episode {episode + 1}: Avg Reward: {avg_reward:.2f}, "
                      f"Avg Steps: {avg_steps:.1f}, ε: {self.epsilon:.3f}")
                
        self.episode_count = num_episodes
        
        if verbose:
            print("Double Q-Learning training completed!")
            
    def get_combined_q_function(self):
        """Get the combined Q-function (Q1 + Q2)"""
        combined_q = defaultdict(lambda: defaultdict(float))
        
        # Combine all states from both Q-functions
        all_states = set(self.Q1.keys()) | set(self.Q2.keys())
        
        for state in all_states:
            for action in Actions:
                combined_q[state][action] = self.Q1[state][action] + self.Q2[state][action]
        
        # Convert to regular dict
        q_dict = {}
        for state in combined_q:
            q_dict[state] = dict(combined_q[state])
        return q_dict
    
    def get_q_functions(self):
        """Get both Q-functions separately"""
        q1_dict = {}
        q2_dict = {}
        
        for state in self.Q1:
            q1_dict[state] = dict(self.Q1[state])
            
        for state in self.Q2:
            q2_dict[state] = dict(self.Q2[state])
            
        return q1_dict, q2_dict
    
    def get_policy(self):
        """Extract greedy policy from combined Q-function"""
        policy = {}
        for state in self.maze.get_all_states():
            if not self.maze.is_terminal(state):
                policy[state] = self.get_greedy_action(state)
        return policy
    
    def get_value_function(self):
        """Extract value function from combined Q-function"""
        V = {}
        for state in self.maze.get_all_states():
            if self.maze.is_terminal(state):
                V[state] = 0.0
            else:
                # V(s) = max_a (Q1(s,a) + Q2(s,a))
                max_q = float('-inf')
                for action in Actions:
                    q_combined = self.get_combined_q_value(state, action)
                    if q_combined > max_q:
                        max_q = q_combined
                V[state] = max_q if max_q != float('-inf') else 0.0
        return V
    
    def get_statistics(self):
        """Get training statistics"""
        return {
            'episodes_trained': self.episode_count,
            'episode_rewards': self.episode_rewards.copy(),
            'learning_curve': self.learning_curve.copy(),
            'episode_lengths': self.episode_lengths.copy(),
            'epsilon_curve': self.epsilon_curve.copy(),
            'final_epsilon': self.epsilon,
            'final_avg_reward': np.mean(self.episode_rewards[-100:]) if len(self.episode_rewards) >= 100 else 0,
            'final_avg_steps': np.mean(self.episode_lengths[-100:]) if len(self.episode_lengths) >= 100 else 0
        }
    
    def evaluate_policy(self, num_episodes=1000, deterministic=True):
        """Evaluate the learned policy"""
        old_epsilon = self.epsilon
        if deterministic:
            self.epsilon = 0.0  # Pure greedy policy
            
        total_rewards = []
        total_steps = []
        
        for _ in range(num_episodes):
            reward, _, steps = self.run_episode()
            total_rewards.append(reward)
            total_steps.append(steps)
            
        self.epsilon = old_epsilon  # Restore original epsilon
        
        return {
            'avg_reward': np.mean(total_rewards),
            'std_reward': np.std(total_rewards),
            'avg_steps': np.mean(total_steps),
            'std_steps': np.std(total_steps),
            'success_rate': sum(1 for r in total_rewards if r > 30) / len(total_rewards)
        }

def compare_double_q_learning_gammas():
    """Compare Double Q-learning with different gamma values"""
    maze = Maze()
    
    print("="*60)
    print("DOUBLE Q-LEARNING ALGORITHM COMPARISON")
    print("="*60)
    
    results = {}
    
    # Double Q-learning with γ = 1.0
    print("\n1. Double Q-Learning with γ = 1.0")
    print("-" * 40)
    double_q_1 = DoubleQLearning(maze, alpha=0.1, gamma=1.0, epsilon=0.1)
    double_q_1.train(num_episodes=8000)
    
    eval_1 = double_q_1.evaluate_policy(num_episodes=1000)
    print(f"Final evaluation: Avg reward: {eval_1['avg_reward']:.2f}, "
          f"Success rate: {eval_1['success_rate']:.2%}")
    
    results['gamma_1.0'] = {
        'double_q': double_q_1,
        'evaluation': eval_1,
        'policy': double_q_1.get_policy(),
        'values': double_q_1.get_value_function(),
        'q_function': double_q_1.get_combined_q_function()
    }
    
    # Double Q-learning with γ = 0.9
    print("\n2. Double Q-Learning with γ = 0.9")
    print("-" * 40)
    double_q_09 = DoubleQLearning(maze, alpha=0.1, gamma=0.9, epsilon=0.1)
    double_q_09.train(num_episodes=8000)
    
    eval_09 = double_q_09.evaluate_policy(num_episodes=1000)
    print(f"Final evaluation: Avg reward: {eval_09['avg_reward']:.2f}, "
          f"Success rate: {eval_09['success_rate']:.2%}")
    
    results['gamma_0.9'] = {
        'double_q': double_q_09,
        'evaluation': eval_09,
        'policy': double_q_09.get_policy(),
        'values': double_q_09.get_value_function(),
        'q_function': double_q_09.get_combined_q_function()
    }
    
    return results

def compare_q_learning_vs_double_q_learning():
    """Compare Q-learning and Double Q-learning performance"""
    from q_learning import QLearning
    
    maze = Maze()
    
    print("="*60)
    print("Q-LEARNING vs DOUBLE Q-LEARNING COMPARISON")
    print("="*60)
    
    # Train both algorithms with same parameters
    print("\nTraining Q-Learning...")
    q_learning = QLearning(maze, alpha=0.1, gamma=0.9, epsilon=0.1)
    q_learning.train(num_episodes=8000, verbose=False)
    q_eval = q_learning.evaluate_policy(num_episodes=1000)
    
    print("\nTraining Double Q-Learning...")
    double_q = DoubleQLearning(maze, alpha=0.1, gamma=0.9, epsilon=0.1)
    double_q.train(num_episodes=8000, verbose=False)
    double_q_eval = double_q.evaluate_policy(num_episodes=1000)
    
    print("\nComparison Results:")
    print("-" * 50)
    print(f"Q-Learning        - Avg Reward: {q_eval['avg_reward']:6.2f}, Success Rate: {q_eval['success_rate']:6.2%}")
    print(f"Double Q-Learning - Avg Reward: {double_q_eval['avg_reward']:6.2f}, Success Rate: {double_q_eval['success_rate']:6.2%}")
    
    # Calculate improvement
    reward_improvement = ((double_q_eval['avg_reward'] - q_eval['avg_reward']) / abs(q_eval['avg_reward'])) * 100
    success_improvement = (double_q_eval['success_rate'] - q_eval['success_rate']) * 100
    
    print(f"\nImprovement with Double Q-Learning:")
    print(f"Reward improvement: {reward_improvement:+.1f}%")
    print(f"Success rate improvement: {success_improvement:+.1f} percentage points")
    
    return {
        'q_learning': {'algorithm': q_learning, 'evaluation': q_eval},
        'double_q_learning': {'algorithm': double_q, 'evaluation': double_q_eval},
        'improvements': {
            'reward': reward_improvement,
            'success_rate': success_improvement
        }
    }

if __name__ == "__main__":
    # Run Double Q-learning experiments
    double_q_results = compare_double_q_learning_gammas()
    
    # Compare Q-learning vs Double Q-learning
    comparison = compare_q_learning_vs_double_q_learning()