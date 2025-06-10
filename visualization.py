# visualization.py
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from maze import Actions
from collections import defaultdict

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def visualize_value_function(maze, value_function, title="Value Function", figsize=(10, 8)):
    """Visualize value function as a heatmap"""

    # Create value grid
    value_grid = np.zeros((maze.height, maze.width))

    for i in range(maze.height):
        for j in range(maze.width):
            state = (i, j)
            value_grid[i, j] = value_function.get(state, 0)

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    im = ax.imshow(value_grid, cmap='RdYlGn', aspect='equal')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('State Value V(s)', rotation=270, labelpad=20)

    # Add text annotations
    for i in range(maze.height):
        for j in range(maze.width):
            state = (i, j)
            value = value_function.get(state, 0)
            reward = maze.get_reward(state)

            # Value text
            ax.text(j, i, f'V={value:.1f}\nR={reward}',
                    ha='center', va='center', fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

            # Mark special states
            if state == maze.start_position:
                ax.text(j, i - 0.4, 'START', ha='center', va='center',
                        fontsize=8, color='purple', weight='bold')

            if maze.is_terminal(state):
                ax.text(j, i - 0.4, 'TERMINAL', ha='center', va='center',
                        fontsize=8, color='darkgreen', weight='bold')

    # Set labels and title
    ax.set_title(title, fontsize=16, pad=20)
    ax.set_xticks(range(maze.width))
    ax.set_yticks(range(maze.height))
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')

    # Add grid
    ax.set_xticks(np.arange(-0.5, maze.width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, maze.height, 1), minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=2)

    plt.tight_layout()
    return fig


def visualize_policy(maze, policy, title="Policy", figsize=(8, 6)):
    """Visualize policy as arrows on grid"""

    action_symbols = {
        Actions.LEFT: '←',
        Actions.UP: '↑',
        Actions.RIGHT: '→',
        Actions.DOWN: '↓',
        None: '•'
    }

    fig, ax = plt.subplots(figsize=figsize)

    # Draw grid
    for i in range(maze.height + 1):
        ax.axhline(i - 0.5, color='black', linewidth=1)
    for j in range(maze.width + 1):
        ax.axvline(j - 0.5, color='black', linewidth=1)

    # Draw policy and rewards
    for i in range(maze.height):
        for j in range(maze.width):
            state = (i, j)
            reward = maze.get_reward(state)

            # Color cell based on reward
            if maze.is_terminal(state):
                color = 'lightgreen' if reward > 0 else 'lightcoral'
                alpha = 0.7
            elif reward < -5:
                color = 'salmon'
                alpha = 0.5
            else:
                color = 'lightblue'
                alpha = 0.3

            # Draw cell background
            rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                 facecolor=color, alpha=alpha)
            ax.add_patch(rect)

            # Add reward text
            ax.text(j, i + 0.3, f'R={reward}', ha='center', va='center',
                    fontsize=10, weight='bold')

            # Add action arrow
            if not maze.is_terminal(state) and state in policy:
                action = policy[state]
                symbol = action_symbols.get(action, '?')
                ax.text(j, i - 0.1, symbol, ha='center', va='center',
                        fontsize=20, color='darkblue', weight='bold')

            # Mark special states
            if state == maze.start_position:
                ax.text(j, i - 0.35, 'START', ha='center', va='center',
                        fontsize=8, color='purple', weight='bold')

    ax.set_xlim(-0.5, maze.width - 0.5)
    ax.set_ylim(-0.5, maze.height - 0.5)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=16)
    ax.set_xticks(range(maze.width))
    ax.set_yticks(range(maze.height))
    ax.invert_yaxis()  # To match array indexing

    plt.tight_layout()
    return fig


def visualize_q_function(maze, q_function, title="Q-Function", figsize=(12, 10)):
    """Visualize Q-function showing Q-values for all state-action pairs"""

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    action_names = ['LEFT', 'UP', 'RIGHT', 'DOWN']
    actions = [Actions.LEFT, Actions.UP, Actions.RIGHT, Actions.DOWN]

    for idx, (action, action_name) in enumerate(zip(actions, action_names)):
        row, col = idx // 2, idx % 2
        ax = axes[row, col]

        # Create Q-value grid for this action
        q_grid = np.zeros((maze.height, maze.width))

        for i in range(maze.height):
            for j in range(maze.width):
                state = (i, j)
                if state in q_function and action in q_function[state]:
                    q_grid[i, j] = q_function[state][action]

        # Create heatmap
        im = ax.imshow(q_grid, cmap='RdYlBu_r', aspect='equal')

        # Add text annotations
        for i in range(maze.height):
            for j in range(maze.width):
                state = (i, j)
                if state in q_function and action in q_function[state]:
                    q_val = q_function[state][action]
                    ax.text(j, i, f'{q_val:.1f}', ha='center', va='center',
                            fontsize=9, weight='bold',
                            color='white' if abs(q_val) > np.max(np.abs(q_grid)) / 2 else 'black')

        ax.set_title(f'Q-values for {action_name}')
        ax.set_xticks(range(maze.width))
        ax.set_yticks(range(maze.height))

        # Add grid
        ax.set_xticks(np.arange(-0.5, maze.width, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, maze.height, 1), minor=True)
        ax.grid(which="minor", color="gray", linestyle='-', linewidth=1)

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    return fig


def plot_learning_curves(algorithms_data, title="Learning Curves", figsize=(12, 8)):
    """Plot learning curves for multiple algorithms"""

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Episode rewards
    ax1 = axes[0, 0]
    for name, data in algorithms_data.items():
        if 'episode_rewards' in data:
            rewards = data['episode_rewards']
            # Smooth with moving average
            window = min(100, len(rewards) // 10)
            if window > 1:
                smoothed = np.convolve(rewards, np.ones(window) / window, mode='valid')
                ax1.plot(range(window - 1, len(rewards)), smoothed, label=name, alpha=0.8)
            else:
                ax1.plot(rewards, label=name, alpha=0.8)

    ax1.set_title('Episode Rewards (Smoothed)')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Episode lengths
    ax2 = axes[0, 1]
    for name, data in algorithms_data.items():
        if 'episode_lengths' in data:
            lengths = data['episode_lengths']
            window = min(100, len(lengths) // 10)
            if window > 1:
                smoothed = np.convolve(lengths, np.ones(window) / window, mode='valid')
                ax2.plot(range(window - 1, len(lengths)), smoothed, label=name, alpha=0.8)
            else:
                ax2.plot(lengths, label=name, alpha=0.8)

    ax2.set_title('Episode Lengths (Smoothed)')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Steps to Terminal')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Learning curve (TD errors)
    ax3 = axes[1, 0]
    for name, data in algorithms_data.items():
        if 'learning_curve' in data:
            errors = data['learning_curve']
            window = min(100, len(errors) // 10)
            if window > 1:
                smoothed = np.convolve(errors, np.ones(window) / window, mode='valid')
                ax3.plot(range(window - 1, len(errors)), smoothed, label=name, alpha=0.8)
            else:
                ax3.plot(errors, label=name, alpha=0.8)

    ax3.set_title('Learning Errors (Smoothed)')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('TD Error')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Epsilon decay
    ax4 = axes[1, 1]
    for name, data in algorithms_data.items():
        if 'epsilon_curve' in data:
            epsilon = data['epsilon_curve']
            ax4.plot(epsilon, label=name, alpha=0.8)

    ax4.set_title('Epsilon Decay')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Exploration Rate (ε)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    return fig


def plot_algorithm_comparison(results_dict, metric='avg_reward', title="Algorithm Comparison"):
    """Create bar plot comparing different algorithms"""

    algorithms = list(results_dict.keys())
    values = []
    errors = []

    for alg in algorithms:
        if 'evaluation' in results_dict[alg]:
            eval_data = results_dict[alg]['evaluation']
            values.append(eval_data.get(metric, 0))
            errors.append(eval_data.get(f'std_{metric.split("_")[-1]}', 0))
        else:
            values.append(0)
            errors.append(0)

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(algorithms, values, yerr=errors, capsize=5, alpha=0.8)

    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + errors[bars.index(bar)],
                f'{value:.2f}', ha='center', va='bottom', fontweight='bold')

    ax.set_title(title, fontsize=14)
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.grid(True, alpha=0.3)

    # Rotate x-axis labels if needed
    if len(max(algorithms, key=len)) > 8:
        plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    return fig


def visualize_path_comparison(maze, policies, figsize=(15, 5)):
    """Visualize paths taken by different policies"""

    n_policies = len(policies)
    fig, axes = plt.subplots(1, n_policies, figsize=figsize)

    if n_policies == 1:
        axes = [axes]

    for idx, (policy_name, policy) in enumerate(policies.items()):
        ax = axes[idx]

        # Draw maze
        for i in range(maze.height + 1):
            ax.axhline(i - 0.5, color='black', linewidth=1)
        for j in range(maze.width + 1):
            ax.axvline(j - 0.5, color='black', linewidth=1)

        # Color cells by reward
        for i in range(maze.height):
            for j in range(maze.width):
                state = (i, j)
                reward = maze.get_reward(state)

                if maze.is_terminal(state):
                    color = 'lightgreen' if reward > 0 else 'lightcoral'
                elif reward < -5:
                    color = 'salmon'
                else:
                    color = 'lightblue'

                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                     facecolor=color, alpha=0.5)
                ax.add_patch(rect)

                ax.text(j, i, f'{reward}', ha='center', va='center',
                        fontsize=8, weight='bold')

        # Trace path
        current_state = maze.start_position
        path = [current_state]
        visited = set()

        while not maze.is_terminal(current_state) and current_state not in visited:
            visited.add(current_state)

            if current_state in policy:
                action = policy[current_state]
                next_state = maze.get_next_state(current_state, action)
                path.append(next_state)
                current_state = next_state
            else:
                break

        # Draw path
        if len(path) > 1:
            path_y = [p[0] for p in path]
            path_x = [p[1] for p in path]
            ax.plot(path_x, path_y, 'ro-', linewidth=3, markersize=8, alpha=0.8)

            # Number the steps
            for step, (x, y) in enumerate(zip(path_x, path_y)):
                ax.text(x + 0.2, y + 0.2, str(step),
                        fontsize=10, weight='bold', color='darkred')

        ax.set_title(f'{policy_name}\nPath Length: {len(path) - 1}')
        ax.set_xlim(-0.5, maze.width - 0.5)
        ax.set_ylim(-0.5, maze.height - 0.5)
        ax.set_aspect('equal')
        ax.set_xticks(range(maze.width))
        ax.set_yticks(range(maze.height))

    plt.tight_layout()
    return fig


def plot_stochastic_comparison(stochastic_results, figsize=(12, 8)):
    """Plot comparison of performance across different noise levels"""

    noise_levels = []
    avg_rewards = []
    success_rates = []
    avg_steps = []

    for key, data in stochastic_results.items():
        if 'noise_level' in data:
            noise_levels.append(data['noise_level'] * 100)  # Convert to percentage
            eval_data = data['evaluation']
            avg_rewards.append(eval_data['avg_reward'])
            success_rates.append(eval_data['success_rate'] * 100)  # Convert to percentage
            avg_steps.append(eval_data['avg_steps'])

    # Sort by noise level
    sorted_data = sorted(zip(noise_levels, avg_rewards, success_rates, avg_steps))
    noise_levels, avg_rewards, success_rates, avg_steps = zip(*sorted_data)

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Average reward vs noise
    axes[0, 0].plot(noise_levels, avg_rewards, 'bo-', linewidth=2, markersize=8)
    axes[0, 0].set_title('Average Reward vs Noise Level')
    axes[0, 0].set_xlabel('Noise Level (%)')
    axes[0, 0].set_ylabel('Average Reward')
    axes[0, 0].grid(True, alpha=0.3)

    # Success rate vs noise
    axes[0, 1].plot(noise_levels, success_rates, 'go-', linewidth=2, markersize=8)
    axes[0, 1].set_title('Success Rate vs Noise Level')
    axes[0, 1].set_xlabel('Noise Level (%)')
    axes[0, 1].set_ylabel('Success Rate (%)')
    axes[0, 1].grid(True, alpha=0.3)

    # Average steps vs noise
    axes[1, 0].plot(noise_levels, avg_steps, 'ro-', linewidth=2, markersize=8)
    axes[1, 0].set_title('Average Steps vs Noise Level')
    axes[1, 0].set_xlabel('Noise Level (%)')
    axes[1, 0].set_ylabel('Average Steps')
    axes[1, 0].grid(True, alpha=0.3)

    # Performance degradation
    if len(avg_rewards) > 1:
        baseline_reward = avg_rewards[0]  # Deterministic performance
        degradation = [(baseline_reward - reward) / abs(baseline_reward) * 100
                       for reward in avg_rewards]

        axes[1, 1].plot(noise_levels, degradation, 'mo-', linewidth=2, markersize=8)
        axes[1, 1].set_title('Performance Degradation')
        axes[1, 1].set_xlabel('Noise Level (%)')
        axes[1, 1].set_ylabel('Reward Degradation (%)')
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def visualize_td_comparison(maze, results_dict, save_prefix="td_learning"):
    """Create comprehensive visualization of TD learning results"""

    # 1. Value Function Heatmaps
    fig1 = plt.figure(figsize=(15, 6))

    for idx, (name, data) in enumerate(results_dict.items()):
        ax = fig1.add_subplot(1, 2, idx + 1)
        values = data['values']

        # Create value grid
        value_grid = np.zeros((maze.height, maze.width))
        for i in range(maze.height):
            for j in range(maze.width):
                state = (i, j)
                value_grid[i, j] = values.get(state, 0)

        # Create heatmap
        im = ax.imshow(value_grid, cmap='RdYlGn', aspect='equal', vmin=-20, vmax=40)

        # Add text annotations
        for i in range(maze.height):
            for j in range(maze.width):
                state = (i, j)
                value = values.get(state, 0)
                reward = maze.get_reward(state)

                # Different formatting for terminal states
                if maze.is_terminal(state):
                    ax.text(j, i, f'TERM\nR={reward}',
                            ha='center', va='center', fontsize=9, weight='bold',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8))
                else:
                    color = 'white' if abs(value) > 10 else 'black'
                    ax.text(j, i, f'V={value:.1f}\nR={reward}',
                            ha='center', va='center', fontsize=8, color=color)

                # Mark start state
                if state == maze.start_position:
                    ax.add_patch(plt.Rectangle((j - 0.45, i - 0.45), 0.9, 0.9,
                                               fill=False, edgecolor='blue', linewidth=3))
                    ax.text(j, i - 0.35, 'START', ha='center', va='center',
                            fontsize=7, color='blue', weight='bold')

        # Formatting
        gamma_val = name.split('_')[1]
        ax.set_title(f'TD Learning Value Function (γ = {gamma_val})', fontsize=14, pad=10)
        ax.set_xticks(range(maze.width))
        ax.set_yticks(range(maze.height))
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
        ax.grid(True, alpha=0.3, linewidth=0.5)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('State Value V(s)', rotation=270, labelpad=15)

    plt.tight_layout()
    return fig1


def create_td_analysis_report(maze, results_dict, save_path="td_analysis_report.txt"):
    """Create detailed analysis report for TD learning results"""

    report = []
    report.append("=" * 80)
    report.append("TEMPORAL DIFFERENCE LEARNING - DETAILED ANALYSIS REPORT")
    report.append("=" * 80)

    # Environment description
    report.append("\n1. ENVIRONMENT DESCRIPTION")
    report.append("-" * 30)
    report.append(f"Grid Size: {maze.height}x{maze.width}")
    report.append(f"Start State: {maze.start_position}")
    report.append(f"Terminal States: {maze.terminal_states}")

    # Results comparison
    report.append("\n2. RESULTS COMPARISON")
    report.append("-" * 30)

    for name, data in results_dict.items():
        gamma = name.split('_')[1]
        stats = data['stats']
        values = data['values']

        report.append(f"\ngamma = {gamma}:")  # Use 'gamma' instead of γ symbol
        report.append(
            f"  - States Discovered: {stats['states_visited']}/{stats['total_states']} ({stats['coverage']:.1%})")
        report.append(f"  - Average Visits per State: {stats['avg_visits']:.1f}")
        report.append(f"  - Final TD Error: {stats['final_avg_error']:.6f}")

        # Value of start state
        start_value = values.get(maze.start_position, 0)
        report.append(f"  - Value of Start State: {start_value:.2f}")

    # Key insights
    report.append("\n3. KEY INSIGHTS")
    report.append("-" * 30)
    report.append("- gamma=1.0: All future rewards valued equally (no discounting)")
    report.append("- gamma=0.5: Future rewards heavily discounted (myopic behavior)")
    report.append("- Random exploration ensures complete state coverage")
    report.append("- TD learning successfully propagates values through bootstrapping")

    # Save report with UTF-8 encoding
    report_text = "\n".join(report)
    with open(save_path, 'w', encoding='utf-8') as f:  # Added UTF-8 encoding
        f.write(report_text)

    print(f"\nDetailed TD analysis report saved to: {save_path}")
    return report_text


def create_summary_report(all_results, save_path=None):
    """Create a comprehensive summary report"""

    report = []
    report.append("=" * 80)
    report.append("MODEL-FREE REINFORCEMENT LEARNING - COMPREHENSIVE REPORT")
    report.append("=" * 80)

    # Add results for each algorithm
    for experiment_name, results in all_results.items():
        report.append(f"\n{experiment_name.upper()}")
        report.append("-" * len(experiment_name))

        if 'evaluation' in results:
            eval_data = results['evaluation']
            report.append(f"Average Reward: {eval_data.get('avg_reward', 'N/A'):.2f}")
            report.append(f"Success Rate: {eval_data.get('success_rate', 'N/A'):.2%}")
            report.append(f"Average Steps: {eval_data.get('avg_steps', 'N/A'):.1f}")

        if 'algorithm' in results and hasattr(results['algorithm'], 'get_statistics'):
            stats = results['algorithm'].get_statistics()
            report.append(f"Episodes Trained: {stats.get('episodes_trained', 'N/A')}")
            report.append(f"Final Epsilon: {stats.get('final_epsilon', 'N/A'):.3f}")

    report_text = "\n".join(report)

    if save_path:
        with open(save_path, 'w') as f:
            f.write(report_text)
        print(f"Report saved to {save_path}")

    return report_text