# stochastic_experiments.py
"""
Part D: Q-learning in stochastic environments
"""
import numpy as np
from maze import Maze
from q_learning import QLearning

def compare_deterministic_vs_stochastic():
    """Compare Q-learning performance in deterministic vs stochastic environments"""
    
    print("="*60)
    print("DETERMINISTIC vs STOCHASTIC ENVIRONMENT COMPARISON")
    print("="*60)
    
    results = {}
    
    # Deterministic environment
    print("\n1. Training Q-Learning in DETERMINISTIC environment")
    print("-" * 50)
    
    det_maze = Maze(stochastic=False)
    det_q_learning = QLearning(det_maze, alpha=0.1, gamma=0.9, epsilon=0.1)
    det_q_learning.train(num_episodes=8000, verbose=True)
    det_eval = det_q_learning.evaluate_policy(num_episodes=1000)
    
    print(f"Deterministic - Final evaluation:")
    print(f"  Avg reward: {det_eval['avg_reward']:.2f}")
    print(f"  Success rate: {det_eval['success_rate']:.2%}")
    print(f"  Avg steps: {det_eval['avg_steps']:.1f}")
    
    results['deterministic'] = {
        'algorithm': det_q_learning,
        'evaluation': det_eval,
        'policy': det_q_learning.get_policy(),
        'values': det_q_learning.get_value_function()
    }
    
    # Stochastic environment (20% noise)
    print("\n2. Training Q-Learning in STOCHASTIC environment (20% noise)")
    print("-" * 50)
    
    stoch_maze = Maze(stochastic=True, noise_prob=0.2)
    stoch_q_learning = QLearning(stoch_maze, alpha=0.1, gamma=0.9, epsilon=0.1)
    stoch_q_learning.train(num_episodes=8000, verbose=True)
    stoch_eval = stoch_q_learning.evaluate_policy(num_episodes=1000)
    
    print(f"Stochastic - Final evaluation:")
    print(f"  Avg reward: {stoch_eval['avg_reward']:.2f}")
    print(f"  Success rate: {stoch_eval['success_rate']:.2%}")
    print(f"  Avg steps: {stoch_eval['avg_steps']:.1f}")
    
    results['stochastic'] = {
        'algorithm': stoch_q_learning,
        'evaluation': stoch_eval,
        'policy': stoch_q_learning.get_policy(),
        'values': stoch_q_learning.get_value_function()
    }
    
    # High noise environment (40% noise)
    print("\n3. Training Q-Learning in HIGH NOISE environment (40% noise)")
    print("-" * 50)
    
    high_noise_maze = Maze(stochastic=True, noise_prob=0.4)
    high_noise_q_learning = QLearning(high_noise_maze, alpha=0.1, gamma=0.9, epsilon=0.1)
    high_noise_q_learning.train(num_episodes=8000, verbose=True)
    high_noise_eval = high_noise_q_learning.evaluate_policy(num_episodes=1000)
    
    print(f"High Noise - Final evaluation:")
    print(f"  Avg reward: {high_noise_eval['avg_reward']:.2f}")
    print(f"  Success rate: {high_noise_eval['success_rate']:.2%}")
    print(f"  Avg steps: {high_noise_eval['avg_steps']:.1f}")
    
    results['high_noise'] = {
        'algorithm': high_noise_q_learning,
        'evaluation': high_noise_eval,
        'policy': high_noise_q_learning.get_policy(),
        'values': high_noise_q_learning.get_value_function()
    }
    
    # Compare results
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    
    environments = ['deterministic', 'stochastic', 'high_noise']
    env_names = ['Deterministic (0% noise)', 'Stochastic (20% noise)', 'High Noise (40% noise)']
    
    print(f"{'Environment':<25} | {'Avg Reward':<12} | {'Success Rate':<13} | {'Avg Steps':<10}")
    print("-" * 70)
    
    for env, name in zip(environments, env_names):
        eval_data = results[env]['evaluation']
        print(f"{name:<25} | {eval_data['avg_reward']:11.2f} | {eval_data['success_rate']:12.2%} | {eval_data['avg_steps']:9.1f}")
    
    # Performance degradation analysis
    det_reward = results['deterministic']['evaluation']['avg_reward']
    stoch_reward = results['stochastic']['evaluation']['avg_reward']
    high_noise_reward = results['high_noise']['evaluation']['avg_reward']
    
    stoch_degradation = ((det_reward - stoch_reward) / abs(det_reward)) * 100
    high_noise_degradation = ((det_reward - high_noise_reward) / abs(det_reward)) * 100
    
    print(f"\nPerformance Degradation from Deterministic:")
    print(f"  20% noise: {stoch_degradation:.1f}% reward decrease")
    print(f"  40% noise: {high_noise_degradation:.1f}% reward decrease")
    
    return results

def analyze_stochastic_learning_curves():
    """Analyze learning curves in different stochastic environments"""
    
    print("\n" + "="*60)
    print("LEARNING CURVE ANALYSIS")
    print("="*60)
    
    noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4]
    results = {}
    
    for noise in noise_levels:
        print(f"\nTraining with {noise*100:.0f}% noise...")
        
        maze = Maze(stochastic=(noise > 0), noise_prob=noise)
        q_learning = QLearning(maze, alpha=0.1, gamma=0.9, epsilon=0.1)
        q_learning.train(num_episodes=5000, verbose=False)
        
        eval_result = q_learning.evaluate_policy(num_episodes=500)
        
        results[f'noise_{noise:.1f}'] = {
            'noise_level': noise,
            'algorithm': q_learning,
            'evaluation': eval_result,
            'learning_curve': q_learning.get_statistics()['episode_rewards']
        }
        
        print(f"  Final avg reward: {eval_result['avg_reward']:.2f}")
        print(f"  Success rate: {eval_result['success_rate']:.2%}")
    
    return results

def test_optimal_hyperparameters_stochastic():
    """Test different hyperparameters for stochastic environments"""
    
    print("\n" + "="*60)
    print("HYPERPARAMETER OPTIMIZATION FOR STOCHASTIC ENVIRONMENTS")
    print("="*60)
    
    # Test different learning rates and exploration rates
    alpha_values = [0.05, 0.1, 0.2, 0.3]
    epsilon_values = [0.05, 0.1, 0.2, 0.3]
    
    best_reward = float('-inf')
    best_params = None
    results = {}
    
    stoch_maze = Maze(stochastic=True, noise_prob=0.2)
    
    print("Testing hyperparameters (this may take a while...):")
    print(f"{'Alpha':<8} | {'Epsilon':<8} | {'Avg Reward':<12} | {'Success Rate':<13}")
    print("-" * 50)
    
    for alpha in alpha_values:
        for epsilon in epsilon_values:
            # Train Q-learning with these parameters
            q_learning = QLearning(stoch_maze, alpha=alpha, gamma=0.9, epsilon=epsilon)
            q_learning.train(num_episodes=4000, verbose=False)
            
            # Evaluate
            evaluation = q_learning.evaluate_policy(num_episodes=500)
            avg_reward = evaluation['avg_reward']
            success_rate = evaluation['success_rate']
            
            print(f"{alpha:<8.2f} | {epsilon:<8.2f} | {avg_reward:<12.2f} | {success_rate:<13.2%}")
            
            results[f'alpha_{alpha}_epsilon_{epsilon}'] = {
                'alpha': alpha,
                'epsilon': epsilon,
                'evaluation': evaluation,
                'algorithm': q_learning
            }
            
            if avg_reward > best_reward:
                best_reward = avg_reward
                best_params = (alpha, epsilon)
    
    print(f"\nBest hyperparameters: α={best_params[0]:.2f}, ε={best_params[1]:.2f}")
    print(f"Best average reward: {best_reward:.2f}")
    
    return results, best_params

if __name__ == "__main__":
    # Run stochastic environment experiments
    print("Running Stochastic Environment Experiments...")
    
    # Basic comparison
    basic_results = compare_deterministic_vs_stochastic()
    
    # Learning curve analysis
    curve_results = analyze_stochastic_learning_curves()
    
    # Hyperparameter optimization
    param_results, best_params = test_optimal_hyperparameters_stochastic()
    
    print("\n" + "="*60)
    print("ALL STOCHASTIC EXPERIMENTS COMPLETED!")
    print("="*60)