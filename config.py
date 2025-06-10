# config.py
"""
Configuration settings for the model-free RL experiments
Modify these parameters to customize the experiments
"""

# Environment Configuration
MAZE_CONFIG = {
    'height': 4,
    'width': 4,
    'start_position': (3, 2),
    'terminal_states': [(0, 3), (3, 0)],
    'stochastic_noise_levels': [0.0, 0.2, 0.4]  # For stochastic experiments
}

# Algorithm Hyperparameters
ALGORITHM_CONFIG = {
    # Temporal Difference Learning
    'td_learning': {
        'alpha': 0.1,           # Learning rate
        'gamma_values': [1.0, 0.5],  # Discount factors to test
        'num_episodes': 5000,   # Training episodes
        'evaluate_all_states': False  # Whether to start from all states
    },
    
    # SARSA
    'sarsa': {
        'alpha': 0.1,           # Learning rate
        'gamma_values': [1.0, 0.9],  # Discount factors to test
        'epsilon': 0.1,         # Initial exploration rate
        'epsilon_decay': 0.995, # Exploration decay rate
        'epsilon_min': 0.01,    # Minimum exploration rate
        'num_episodes': 8000,   # Training episodes
        'eval_episodes': 1000   # Evaluation episodes
    },
    
    # Q-Learning
    'q_learning': {
        'alpha': 0.1,           # Learning rate
        'gamma_values': [1.0, 0.9],  # Discount factors to test
        'epsilon': 0.1,         # Initial exploration rate
        'epsilon_decay': 0.995, # Exploration decay rate
        'epsilon_min': 0.01,    # Minimum exploration rate
        'num_episodes': 8000,   # Training episodes
        'eval_episodes': 1000   # Evaluation episodes
    },
    
    # Double Q-Learning
    'double_q_learning': {
        'alpha': 0.1,           # Learning rate
        'gamma_values': [1.0, 0.9],  # Discount factors to test
        'epsilon': 0.1,         # Initial exploration rate
        'epsilon_decay': 0.995, # Exploration decay rate
        'epsilon_min': 0.01,    # Minimum exploration rate
        'num_episodes': 8000,   # Training episodes
        'eval_episodes': 1000   # Evaluation episodes
    }
}

# Stochastic Environment Configuration
STOCHASTIC_CONFIG = {
    'noise_levels': [0.0, 0.1, 0.2, 0.3, 0.4],  # Noise probability levels
    'num_episodes': 5000,       # Episodes per noise level
    'eval_episodes': 500,       # Evaluation episodes
    'hyperparameter_search': {
        'alpha_values': [0.05, 0.1, 0.2, 0.3],
        'epsilon_values': [0.05, 0.1, 0.2, 0.3],
        'test_episodes': 4000   # Episodes for hyperparameter testing
    }
}

# Visualization Configuration
VISUALIZATION_CONFIG = {
    'figure_size': {
        'value_function': (10, 8),
        'policy': (8, 6),
        'q_function': (12, 10),
        'learning_curves': (12, 8),
        'comparison': (10, 6),
        'path_comparison': (15, 5)
    },
    'dpi': 300,                 # Figure resolution
    'style': 'seaborn-v0_8',    # Matplotlib style
    'color_palette': 'husl',    # Seaborn color palette
    'save_format': 'png'        # Figure save format
}

# Experiment Configuration
EXPERIMENT_CONFIG = {
    'run_td_learning': True,        # Part A
    'run_sarsa': True,              # Part B
    'run_q_learning': True,         # Part C
    'run_stochastic': True,         # Part D
    'run_double_q_learning': True,  # Part E
    'run_comparisons': True,        # Algorithm comparisons
    'verbose': True,                # Print progress
    'save_intermediate': True,      # Save intermediate results
    'generate_report': True         # Generate text report
}

# Performance Thresholds (for testing)
PERFORMANCE_THRESHOLDS = {
    'min_avg_reward': 20,       # Minimum acceptable average reward
    'min_success_rate': 0.7,    # Minimum acceptable success rate
    'max_avg_steps': 50,        # Maximum acceptable average steps
    'convergence_window': 100   # Episodes to check for convergence
}

# File Output Configuration
OUTPUT_CONFIG = {
    'create_timestamp_dir': True,   # Create timestamped output directory
    'save_figures': True,           # Save visualization figures
    'save_data': True,              # Save experiment data
    'save_report': True,            # Save text report
    'output_prefix': 'model_free_rl'  # Prefix for output files
}

def get_algorithm_config(algorithm_name):
    """Get configuration for a specific algorithm"""
    return ALGORITHM_CONFIG.get(algorithm_name, {})

def get_gamma_values(algorithm_name):
    """Get gamma values to test for a specific algorithm"""
    config = get_algorithm_config(algorithm_name)
    return config.get('gamma_values', [1.0])

def get_training_episodes(algorithm_name):
    """Get number of training episodes for a specific algorithm"""
    config = get_algorithm_config(algorithm_name)
    return config.get('num_episodes', 5000)

def get_figure_size(plot_type):
    """Get figure size for a specific plot type"""
    return VISUALIZATION_CONFIG['figure_size'].get(plot_type, (10, 6))

# Quick access to commonly used values
DEFAULT_ALPHA = 0.1
DEFAULT_GAMMA = 0.9
DEFAULT_EPSILON = 0.1
DEFAULT_EPISODES = 8000
DEFAULT_EVAL_EPISODES = 1000

# Print configuration summary
def print_config_summary():
    """Print a summary of the current configuration"""
    print("="*60)
    print("EXPERIMENT CONFIGURATION SUMMARY")
    print("="*60)
    
    print(f"Environment: {MAZE_CONFIG['height']}x{MAZE_CONFIG['width']} maze")
    print(f"Start position: {MAZE_CONFIG['start_position']}")
    print(f"Terminal states: {MAZE_CONFIG['terminal_states']}")
    
    print(f"\nAlgorithms to run:")
    for name, enabled in EXPERIMENT_CONFIG.items():
        if name.startswith('run_') and enabled:
            algo_name = name[4:].replace('_', ' ').title()
            print(f"  ✓ {algo_name}")
    
    print(f"\nDefault hyperparameters:")
    print(f"  Learning rate (α): {DEFAULT_ALPHA}")
    print(f"  Discount factor (γ): {DEFAULT_GAMMA}")
    print(f"  Exploration rate (ε): {DEFAULT_EPSILON}")
    print(f"  Training episodes: {DEFAULT_EPISODES}")
    
    print("="*60)

if __name__ == "__main__":
    print_config_summary()