# sarsa.py
import numpy as np
import random
from collections import defaultdict
from maze import Maze, Actions


class SARSA:
    """SARSA: On-policy TD Control Algorithm with Exploration"""

    def __init__(self, maze, alpha=0.1, gamma=1.0, epsilon=1.0, epsilon_decay=0.999,
                 epsilon_min=0.01, optimistic_init=2.0):
        self.maze = maze
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate (start high!)
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.optimistic_init = optimistic_init  # Optimistic initialization value

        # Initialize Q-function with optimistic values to encourage exploration
        self.Q = defaultdict(lambda: defaultdict(lambda: optimistic_init))

        # Initialize Q(terminal, ·) = 0 for all actions
        for terminal_state in maze.terminal_states:
            for action in Actions:
                self.Q[terminal_state][action] = 0.0

        # Statistics
        self.episode_count = 0
        self.learning_curve = []
        self.epsilon_curve = []
        self.episode_rewards = []
        self.episode_lengths = []
        self.td_errors = []  # Track TD errors
        self.state_visits = defaultdict(int)  # Track state visitation

    def get_epsilon_greedy_action(self, state):
        """Choose action using ε-greedy policy"""
        if self.maze.is_terminal(state):
            return None

        # ε-greedy action selection
        if random.random() < self.epsilon:
            # Explore: choose random action
            return random.choice(list(Actions))
        else:
            # Exploit: choose best action
            return self.get_greedy_action(state)

    def get_greedy_action(self, state):
        """Get greedy action (best Q-value) for given state"""
        if self.maze.is_terminal(state):
            return None

        # Find action with highest Q-value
        best_actions = []
        best_value = float('-inf')

        for action in Actions:
            q_value = self.Q[state][action]
            if q_value > best_value:
                best_value = q_value
                best_actions = [action]
            elif q_value == best_value:
                best_actions.append(action)

        # Break ties randomly
        return random.choice(best_actions)

    def sarsa_update(self, state, action, reward, next_state, next_action):
        """Perform SARSA update"""
        if not self.maze.is_terminal(state):
            # SARSA update: Q(S,A) ← Q(S,A) + α[R + γQ(S',A') - Q(S,A)]
            if next_action is None:  # Next state is terminal
                q_next = 0.0
            else:
                q_next = self.Q[next_state][next_action]

            q_target = reward + self.gamma * q_next
            q_error = q_target - self.Q[state][action]
            self.Q[state][action] += self.alpha * q_error

            self.td_errors.append(abs(q_error))
            return abs(q_error)
        return 0

    def run_episode(self, max_steps=200, start_state=None):
        """Run one episode using SARSA"""
        if start_state is None:
            state = self.maze.reset()
        else:
            state = start_state

        action = self.get_epsilon_greedy_action(state)

        total_reward = 0
        total_error = 0
        steps = 0
        episode_states = [state]

        while not self.maze.is_terminal(state) and steps < max_steps:
            # Track state visits
            self.state_visits[state] += 1

            # Take action A, observe R, S'
            next_state, reward, done = self.maze.step(state, action)
            total_reward += reward
            episode_states.append(next_state)

            # Choose A' from S' using policy derived from Q (ε-greedy)
            next_action = self.get_epsilon_greedy_action(next_state)

            # SARSA update
            error = self.sarsa_update(state, action, reward, next_state, next_action)
            total_error += error

            # Move to next state and action
            state = next_state
            action = next_action
            steps += 1

        # Record which terminal was reached
        if self.maze.is_terminal(state):
            self.state_visits[state] += 1

        return total_reward, total_error, steps, state

    def train(self, num_episodes=15000, verbose=True, exploration_boost_episodes=5000):
        """Train using SARSA algorithm with exploration boost phase"""
        if verbose:
            print(f"Training SARSA with α={self.alpha}, γ={self.gamma}")
            print(f"Initial ε={self.epsilon}, decay={self.epsilon_decay}, min={self.epsilon_min}")
            print(f"Optimistic initialization: {self.optimistic_init}")
            print(f"Exploration boost for first {exploration_boost_episodes} episodes")

        terminals_reached = {(0, 3): 0, (3, 0): 0}  # Track which terminals are found

        for episode in range(num_episodes):
            # Enhanced exploration for early episodes
            if episode < exploration_boost_episodes:
                # Occasionally start from random positions to ensure exploration
                if episode % 10 == 0 and random.random() < 0.3:
                    all_states = [s for s in self.maze.get_all_states()
                                  if not self.maze.is_terminal(s)]
                    start_state = random.choice(all_states)
                else:
                    start_state = None
            else:
                start_state = None

            # Run episode
            reward, error, steps, final_state = self.run_episode(start_state=start_state)

            # Track which terminal was reached
            if final_state in terminals_reached:
                terminals_reached[final_state] += 1

            # Store statistics
            self.episode_rewards.append(reward)
            self.learning_curve.append(error)
            self.episode_lengths.append(steps)
            self.epsilon_curve.append(self.epsilon)

            # Adaptive epsilon decay
            if episode < exploration_boost_episodes:
                # Slower decay during exploration phase
                if self.epsilon > 0.1:
                    self.epsilon *= (self.epsilon_decay ** 0.5)  # Slower decay
            else:
                # Normal decay after exploration phase
                if self.epsilon > self.epsilon_min:
                    self.epsilon *= self.epsilon_decay

            # Verbose output
            if verbose and (episode + 1) % 2000 == 0:
                avg_reward = np.mean(self.episode_rewards[-500:])
                avg_steps = np.mean(self.episode_lengths[-500:])
                recent_terminals = {
                    (0, 3): sum(1 for r in self.episode_rewards[-500:] if r > 30),
                    (3, 0): sum(1 for r in self.episode_rewards[-500:] if 5 < r < 15)
                }

                print(f"Episode {episode + 1}:")
                print(f"  Avg Reward: {avg_reward:.2f}, Avg Steps: {avg_steps:.1f}, ε: {self.epsilon:.3f}")
                print(f"  Terminals reached (last 500): +40 terminal: {recent_terminals[(0, 3)]}, "
                      f"+10 terminal: {recent_terminals[(3, 0)]}")
                print(f"  State coverage: {len(self.state_visits)}/{len(self.maze.get_all_states())} states")

        self.episode_count = num_episodes

        if verbose:
            print("\nSARSA training completed!")
            print(f"Total terminals reached: +40: {terminals_reached[(0, 3)]}, "
                  f"+10: {terminals_reached[(3, 0)]}")
            print(f"Final state coverage: {len(self.state_visits)}/{len(self.maze.get_all_states())} states")

    def get_q_function(self):
        """Get the learned Q-function"""
        q_dict = {}
        for state in self.Q:
            q_dict[state] = dict(self.Q[state])
        return q_dict

    def get_policy(self):
        """Extract greedy policy from Q-function"""
        policy = {}
        for state in self.maze.get_all_states():
            if not self.maze.is_terminal(state):
                policy[state] = self.get_greedy_action(state)
        return policy

    def get_value_function(self):
        """Extract value function from Q-function (V(s) = max_a Q(s,a))"""
        V = {}
        for state in self.maze.get_all_states():
            if self.maze.is_terminal(state):
                V[state] = 0.0
            else:
                if state in self.Q and self.Q[state]:
                    V[state] = max(self.Q[state].values())
                else:
                    V[state] = 0.0
        return V

    def get_statistics(self):
        """Get training statistics"""
        # Calculate moving averages for smoother curves
        window = 100
        smoothed_rewards = []
        smoothed_lengths = []

        for i in range(len(self.episode_rewards)):
            start_idx = max(0, i - window + 1)
            smoothed_rewards.append(np.mean(self.episode_rewards[start_idx:i + 1]))
            smoothed_lengths.append(np.mean(self.episode_lengths[start_idx:i + 1]))

        return {
            'episodes_trained': self.episode_count,
            'episode_rewards': self.episode_rewards.copy(),
            'learning_curve': self.learning_curve.copy(),
            'episode_lengths': self.episode_lengths.copy(),
            'epsilon_curve': self.epsilon_curve.copy(),
            'final_epsilon': self.epsilon,
            'final_avg_reward': np.mean(self.episode_rewards[-500:]) if len(self.episode_rewards) >= 500 else 0,
            'final_avg_steps': np.mean(self.episode_lengths[-500:]) if len(self.episode_lengths) >= 500 else 0,
            'td_errors': self.td_errors.copy(),
            'state_visits': dict(self.state_visits),
            'smoothed_rewards': smoothed_rewards,
            'smoothed_lengths': smoothed_lengths
        }

    def evaluate_policy(self, num_episodes=1000, deterministic=True):
        """Evaluate the learned policy"""
        old_epsilon = self.epsilon
        if deterministic:
            self.epsilon = 0.0  # Pure greedy policy

        total_rewards = []
        total_steps = []
        terminals_reached = {(0, 3): 0, (3, 0): 0}

        for _ in range(num_episodes):
            reward, _, steps, final_state = self.run_episode()
            total_rewards.append(reward)
            total_steps.append(steps)
            if final_state in terminals_reached:
                terminals_reached[final_state] += 1

        self.epsilon = old_epsilon  # Restore original epsilon

        # Success rate: reaching the +40 terminal
        success_rate = terminals_reached[(0, 3)] / num_episodes

        return {
            'avg_reward': np.mean(total_rewards),
            'std_reward': np.std(total_rewards),
            'avg_steps': np.mean(total_steps),
            'std_steps': np.std(total_steps),
            'success_rate': success_rate,
            'terminals_reached': terminals_reached
        }


def compare_sarsa_gammas():
    """Compare SARSA with different gamma values"""
    maze = Maze()

    print("=" * 60)
    print("SARSA ALGORITHM COMPARISON")
    print("=" * 60)

    results = {}

    # SARSA with γ = 1.0
    print("\n1. SARSA with γ = 1.0 (No Discounting)")
    print("-" * 50)
    sarsa_1 = SARSA(maze, alpha=0.1, gamma=1.0, epsilon=1.0,
                    epsilon_decay=0.999, epsilon_min=0.01, optimistic_init=5.0)
    sarsa_1.train(num_episodes=15000, exploration_boost_episodes=5000)

    eval_1 = sarsa_1.evaluate_policy(num_episodes=1000)
    print(f"\nFinal evaluation:")
    print(f"  Average reward: {eval_1['avg_reward']:.2f} ± {eval_1['std_reward']:.2f}")
    print(f"  Success rate (reaching +40): {eval_1['success_rate']:.2%}")
    print(f"  Average steps: {eval_1['avg_steps']:.1f}")

    results['gamma_1.0'] = {
        'sarsa': sarsa_1,
        'evaluation': eval_1,
        'policy': sarsa_1.get_policy(),
        'values': sarsa_1.get_value_function(),
        'q_function': sarsa_1.get_q_function()
    }

    # SARSA with γ = 0.9
    print("\n2. SARSA with γ = 0.9 (Moderate Discounting)")
    print("-" * 50)
    sarsa_09 = SARSA(maze, alpha=0.1, gamma=0.9, epsilon=1.0,
                     epsilon_decay=0.999, epsilon_min=0.01, optimistic_init=5.0)
    sarsa_09.train(num_episodes=15000, exploration_boost_episodes=5000)

    eval_09 = sarsa_09.evaluate_policy(num_episodes=1000)
    print(f"\nFinal evaluation:")
    print(f"  Average reward: {eval_09['avg_reward']:.2f} ± {eval_09['std_reward']:.2f}")
    print(f"  Success rate (reaching +40): {eval_09['success_rate']:.2%}")
    print(f"  Average steps: {eval_09['avg_steps']:.1f}")

    results['gamma_0.9'] = {
        'sarsa': sarsa_09,
        'evaluation': eval_09,
        'policy': sarsa_09.get_policy(),
        'values': sarsa_09.get_value_function(),
        'q_function': sarsa_09.get_q_function()
    }

    # Analysis comparison
    print("\n3. SARSA Characteristics Analysis")
    print("-" * 50)
    print("Key differences between γ=1.0 and γ=0.9:")

    v1_start = results['gamma_1.0']['values'].get((3, 2), 0)
    v09_start = results['gamma_0.9']['values'].get((3, 2), 0)

    print(f"  Start state values: γ=1.0: {v1_start:.2f}, γ=0.9: {v09_start:.2f}")
    print(f"  Performance gap: {eval_1['avg_reward'] - eval_09['avg_reward']:.2f}")

    # Check if policies differ
    policy_1 = results['gamma_1.0']['policy']
    policy_09 = results['gamma_0.9']['policy']
    policy_diffs = sum(1 for s in policy_1 if s in policy_09 and policy_1[s] != policy_09[s])
    print(f"  Policy differences: {policy_diffs} states with different actions")

    return results


def demonstrate_sarsa_characteristics(maze):
    """Demonstrate key SARSA characteristics for educational purposes"""

    print("\n" + "=" * 60)
    print("SARSA CHARACTERISTICS DEMONSTRATION")
    print("=" * 60)

    # 1. On-policy nature demonstration
    print("\n1. ON-POLICY LEARNING")
    print("-" * 30)
    print("SARSA learns about the policy it follows (including exploration)")
    print("This means Q-values reflect the actual behavior policy, not just optimal actions")

    # Initialize SARSA with high exploration
    sarsa_explore = SARSA(maze, alpha=0.5, gamma=0.9, epsilon=0.5, epsilon_decay=1.0)  # Fixed epsilon

    # Run a few episodes to show on-policy updates
    print("\nRunning 5 episodes with ε=0.5 (high exploration):")
    for ep in range(5):
        state = maze.reset()
        action = sarsa_explore.get_epsilon_greedy_action(state)
        print(f"\nEpisode {ep + 1}:")

        for step in range(3):  # Just show first 3 steps
            if maze.is_terminal(state):
                break

            next_state, reward, done = maze.step(state, action)
            next_action = sarsa_explore.get_epsilon_greedy_action(next_state)

            # Show the on-policy update
            if next_action:
                q_next = sarsa_explore.Q[next_state][next_action]
            else:
                q_next = 0

            old_q = sarsa_explore.Q[state][action]
            target = reward + sarsa_explore.gamma * q_next

            print(f"  Step {step + 1}: {state} --{action.name}--> {next_state}")
            print(f"    Reward: {reward}, Next action: {next_action.name if next_action else 'Terminal'}")
            print(f"    Q({state}, {action.name}): {old_q:.2f} -> {old_q + sarsa_explore.alpha * (target - old_q):.2f}")

            sarsa_explore.sarsa_update(state, action, reward, next_state, next_action)
            state = next_state
            action = next_action

    # 2. Conservative behavior
    print("\n2. CONSERVATIVE BEHAVIOR")
    print("-" * 30)
    print("SARSA tends to learn safer policies due to on-policy learning")
    print("It accounts for exploration mistakes in its value estimates")

    return sarsa_explore


if __name__ == "__main__":
    # Run main comparison
    results = compare_sarsa_gammas()

    # Demonstrate SARSA characteristics
    maze = Maze()
    demonstrate_sarsa_characteristics(maze)