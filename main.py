# main.py
"""
Model-Free Reinforcement Learning Assignment
Main script to run all experiments and generate comprehensive results
"""

import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

# Import all our modules
from maze import Maze
from td_learning import compare_td_methods, OptimalPolicy
from sarsa import compare_sarsa_gammas
from q_learning import compare_q_learning_gammas, compare_sarsa_vs_q_learning
from double_q_learning import compare_double_q_learning_gammas, compare_q_learning_vs_double_q_learning
from stochastic_experiments import compare_deterministic_vs_stochastic, analyze_stochastic_learning_curves
from visualization import *


def create_output_directory():
    """Create output directory for results"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def save_figure(fig, filename, output_dir):
    """Save figure to output directory"""
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {filepath}")


def run_part_a_td_learning(output_dir):
    """Part A: Temporal Difference Learning"""
    print("\n" + "=" * 80)
    print("PART A: TEMPORAL DIFFERENCE LEARNING")
    print("=" * 80)

    # Run improved TD learning experiments
    td_results = compare_td_methods()

    # Create enhanced visualizations
    maze = Maze()
    fig1 = visualize_td_comparison(maze, td_results, save_prefix="td_learning")
    save_figure(fig1, "td_learning_enhanced_comparison.png", output_dir)

    # Create detailed analysis report
    report_path = os.path.join(output_dir, "td_analysis_report.txt")
    create_td_analysis_report(maze, td_results, save_path=report_path)

    # Standard visualizations for compatibility
    for gamma_str, data in td_results.items():
        gamma_val = float(gamma_str.split('_')[1])
        values = data['values']

        # Value function visualization
        fig = visualize_value_function(
            Maze(), values,
            title=f"TD Learning Value Function (γ={gamma_val})"
        )
        save_figure(fig, f"td_values_gamma_{gamma_val}.png", output_dir)

    # Plot learning curves comparison
    td_stats = {}
    for gamma_str, data in td_results.items():
        td_stats[f"TD {gamma_str}"] = data['stats']

    fig = plot_learning_curves(td_stats, "TD Learning - Enhanced Learning Curves")
    save_figure(fig, "td_learning_curves.png", output_dir)

    # Print summary of improvements
    print(f"\n🎯 TD Learning Results Summary:")
    for gamma_str, data in td_results.items():
        gamma_val = gamma_str.split('_')[1]
        stats = data['stats']
        values = data['values']
        start_value = values.get(maze.start_position, 0)

        print(f"   γ={gamma_val}: Start state V = {start_value:.2f}, "
              f"Coverage = {stats['coverage']:.1%}, "
              f"Avg visits = {stats['avg_visits']:.1f}")

    return td_results


def run_part_b_sarsa(output_dir):
    """Part B: SARSA"""
    print("\n" + "=" * 80)
    print("PART B: SARSA (ON-POLICY TD CONTROL)")
    print("=" * 80)

    # Run SARSA experiments
    sarsa_results = compare_sarsa_gammas()

    # Visualize results for each gamma
    for gamma_str, data in sarsa_results.items():
        gamma_val = float(gamma_str.split('_')[1])

        # Value function
        fig = visualize_value_function(
            Maze(), data['values'],
            title=f"SARSA Value Function (γ={gamma_val})"
        )
        save_figure(fig, f"sarsa_values_gamma_{gamma_val}.png", output_dir)

        # Policy
        fig = visualize_policy(
            Maze(), data['policy'],
            title=f"SARSA Policy (γ={gamma_val})"
        )
        save_figure(fig, f"sarsa_policy_gamma_{gamma_val}.png", output_dir)

        # Q-function
        fig = visualize_q_function(
            Maze(), data['q_function'],
            title=f"SARSA Q-Function (γ={gamma_val})"
        )
        save_figure(fig, f"sarsa_q_function_gamma_{gamma_val}.png", output_dir)

    # Learning curves
    sarsa_stats = {}
    for gamma_str, data in sarsa_results.items():
        sarsa_stats[f"SARSA {gamma_str}"] = data['sarsa'].get_statistics()

    fig = plot_learning_curves(sarsa_stats, "SARSA - Learning Curves")
    save_figure(fig, "sarsa_learning_curves.png", output_dir)

    return sarsa_results


def run_part_c_q_learning(output_dir):
    """Part C: Q-Learning"""
    print("\n" + "=" * 80)
    print("PART C: Q-LEARNING (OFF-POLICY TD CONTROL)")
    print("=" * 80)

    # Run Q-learning experiments
    q_results = compare_q_learning_gammas()

    # Visualize results for each gamma
    for gamma_str, data in q_results.items():
        gamma_val = float(gamma_str.split('_')[1])

        # Value function
        fig = visualize_value_function(
            Maze(), data['values'],
            title=f"Q-Learning Value Function (γ={gamma_val})"
        )
        save_figure(fig, f"q_learning_values_gamma_{gamma_val}.png", output_dir)

        # Policy
        fig = visualize_policy(
            Maze(), data['policy'],
            title=f"Q-Learning Policy (γ={gamma_val})"
        )
        save_figure(fig, f"q_learning_policy_gamma_{gamma_val}.png", output_dir)

        # Q-function
        fig = visualize_q_function(
            Maze(), data['q_function'],
            title=f"Q-Learning Q-Function (γ={gamma_val})"
        )
        save_figure(fig, f"q_learning_q_function_gamma_{gamma_val}.png", output_dir)

    # Learning curves
    q_stats = {}
    for gamma_str, data in q_results.items():
        q_stats[f"Q-Learning {gamma_str}"] = data['q_learning'].get_statistics()

    fig = plot_learning_curves(q_stats, "Q-Learning - Learning Curves")
    save_figure(fig, "q_learning_curves.png", output_dir)

    return q_results


def run_part_d_stochastic(output_dir):
    """Part D: Stochastic Environment"""
    print("\n" + "=" * 80)
    print("PART D: Q-LEARNING IN STOCHASTIC ENVIRONMENTS")
    print("=" * 80)

    # Run stochastic experiments
    stoch_results = compare_deterministic_vs_stochastic()

    # Visualize policies for different noise levels
    policies = {}
    for env_name, data in stoch_results.items():
        policies[f"{env_name.title()} Policy"] = data['policy']

    fig = visualize_path_comparison(Maze(), policies)
    save_figure(fig, "stochastic_policies_comparison.png", output_dir)

    # Performance comparison
    comparison_data = {}
    for env_name, data in stoch_results.items():
        comparison_data[env_name.replace('_', ' ').title()] = data

    fig = plot_algorithm_comparison(
        comparison_data, 'avg_reward',
        "Performance vs Environment Stochasticity"
    )
    save_figure(fig, "stochastic_performance_comparison.png", output_dir)

    # Learning curve analysis
    curve_results = analyze_stochastic_learning_curves()
    fig = plot_stochastic_comparison(curve_results)
    save_figure(fig, "stochastic_noise_analysis.png", output_dir)

    return stoch_results, curve_results


def run_part_e_double_q_learning(output_dir):
    """Part E: Double Q-Learning"""
    print("\n" + "=" * 80)
    print("PART E: DOUBLE Q-LEARNING")
    print("=" * 80)

    # Run Double Q-learning experiments
    double_q_results = compare_double_q_learning_gammas()

    # Visualize results for each gamma
    for gamma_str, data in double_q_results.items():
        gamma_val = float(gamma_str.split('_')[1])

        # Value function
        fig = visualize_value_function(
            Maze(), data['values'],
            title=f"Double Q-Learning Value Function (γ={gamma_val})"
        )
        save_figure(fig, f"double_q_values_gamma_{gamma_val}.png", output_dir)

        # Policy
        fig = visualize_policy(
            Maze(), data['policy'],
            title=f"Double Q-Learning Policy (γ={gamma_val})"
        )
        save_figure(fig, f"double_q_policy_gamma_{gamma_val}.png", output_dir)

    # Learning curves
    double_q_stats = {}
    for gamma_str, data in double_q_results.items():
        double_q_stats[f"Double Q-Learning {gamma_str}"] = data['double_q'].get_statistics()

    fig = plot_learning_curves(double_q_stats, "Double Q-Learning - Learning Curves")
    save_figure(fig, "double_q_learning_curves.png", output_dir)

    return double_q_results


def run_algorithm_comparisons(output_dir, sarsa_results, q_results, double_q_results):
    """Compare all algorithms"""
    print("\n" + "=" * 80)
    print("ALGORITHM COMPARISONS")
    print("=" * 80)

    # SARSA vs Q-Learning
    comparison_1 = compare_sarsa_vs_q_learning()

    # Q-Learning vs Double Q-Learning
    comparison_2 = compare_q_learning_vs_double_q_learning()

    # Overall comparison (gamma = 0.9)
    overall_comparison = {
        'SARSA': sarsa_results['gamma_0.9'],
        'Q-Learning': q_results['gamma_0.9'],
        'Double Q-Learning': double_q_results['gamma_0.9']
    }

    # Performance comparison chart
    fig = plot_algorithm_comparison(
        overall_comparison, 'avg_reward',
        "Algorithm Performance Comparison (γ=0.9)"
    )
    save_figure(fig, "overall_algorithm_comparison.png", output_dir)

    # Success rate comparison
    fig = plot_algorithm_comparison(
        overall_comparison, 'success_rate',
        "Algorithm Success Rate Comparison (γ=0.9)"
    )
    save_figure(fig, "algorithm_success_rate_comparison.png", output_dir)

    # Learning curves comparison
    all_stats = {}
    for name, data in overall_comparison.items():
        if 'sarsa' in data:
            all_stats[name] = data['sarsa'].get_statistics()
        elif 'q_learning' in data:
            all_stats[name] = data['q_learning'].get_statistics()
        elif 'double_q' in data:
            all_stats[name] = data['double_q'].get_statistics()

    fig = plot_learning_curves(all_stats, "All Algorithms - Learning Curves Comparison")
    save_figure(fig, "all_algorithms_learning_curves.png", output_dir)

    return comparison_1, comparison_2, overall_comparison


def generate_comprehensive_report(output_dir, all_results):
    """Generate a comprehensive text report"""

    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("MODEL-FREE REINFORCEMENT LEARNING - COMPREHENSIVE REPORT")
    report_lines.append("=" * 80)
    report_lines.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")

    # Summary of experiments
    report_lines.append("EXPERIMENTS CONDUCTED:")
    report_lines.append("-" * 40)
    report_lines.append("A. Temporal Difference Learning (TD(0))")
    report_lines.append("B. SARSA (On-policy TD Control)")
    report_lines.append("C. Q-Learning (Off-policy TD Control)")
    report_lines.append("D. Stochastic Environment Q-Learning")
    report_lines.append("E. Double Q-Learning")
    report_lines.append("")

    # Key findings
    report_lines.append("KEY FINDINGS:")
    report_lines.append("-" * 40)

    # Add algorithm performance summary
    if 'overall_comparison' in all_results:
        for alg_name, data in all_results['overall_comparison'].items():
            if 'evaluation' in data:
                eval_data = data['evaluation']
                report_lines.append(f"{alg_name}:")
                report_lines.append(f"  - Average Reward: {eval_data['avg_reward']:.2f}")
                report_lines.append(f"  - Success Rate: {eval_data['success_rate']:.2%}")
                report_lines.append(f"  - Average Steps: {eval_data['avg_steps']:.1f}")
                report_lines.append("")

    # Theoretical insights
    report_lines.append("THEORETICAL INSIGHTS:")
    report_lines.append("-" * 40)
    report_lines.append("1. TD Learning vs Monte Carlo:")
    report_lines.append("   - TD learns online, MC waits for episode completion")
    report_lines.append("   - TD has bias but lower variance")
    report_lines.append("")
    report_lines.append("2. On-policy (SARSA) vs Off-policy (Q-Learning):")
    report_lines.append("   - SARSA learns about the policy it follows")
    report_lines.append("   - Q-Learning learns about the optimal policy")
    report_lines.append("   - Q-Learning typically converges faster to optimal")
    report_lines.append("")
    report_lines.append("3. Double Q-Learning:")
    report_lines.append("   - Reduces overestimation bias present in Q-Learning")
    report_lines.append("   - Uses two Q-tables to separate action selection and evaluation")
    report_lines.append("")
    report_lines.append("4. Stochastic Environments:")
    report_lines.append("   - Performance degrades with increasing noise")
    report_lines.append("   - Requires more exploration and robust policies")
    report_lines.append("")

    # Files generated
    report_lines.append("GENERATED FILES:")
    report_lines.append("-" * 40)
    for file in os.listdir(output_dir):
        if file.endswith('.png'):
            report_lines.append(f"  - {file}")

    # Save report
    report_text = "\n".join(report_lines)
    report_path = os.path.join(output_dir, "comprehensive_report.txt")

    with open(report_path, 'w') as f:
        f.write(report_text)

    print(f"\nComprehensive report saved to: {report_path}")
    return report_text


def main():
    """Run all experiments and generate comprehensive results"""

    print("=" * 80)
    print("MODEL-FREE REINFORCEMENT LEARNING ASSIGNMENT")
    print("Running comprehensive experiments...")
    print("=" * 80)

    # Create output directory
    output_dir = create_output_directory()
    print(f"Results will be saved to: {output_dir}")

    # Store all results
    all_results = {}

    try:
        # Part A: TD Learning
        td_results = run_part_a_td_learning(output_dir)
        all_results['td_learning'] = td_results

        # Part B: SARSA
        sarsa_results = run_part_b_sarsa(output_dir)
        all_results['sarsa'] = sarsa_results

        # Part C: Q-Learning
        q_results = run_part_c_q_learning(output_dir)
        all_results['q_learning'] = q_results

        # Part D: Stochastic Environment
        stoch_results, curve_results = run_part_d_stochastic(output_dir)
        all_results['stochastic'] = stoch_results
        all_results['stochastic_curves'] = curve_results

        # Part E: Double Q-Learning
        double_q_results = run_part_e_double_q_learning(output_dir)
        all_results['double_q_learning'] = double_q_results

        # Algorithm Comparisons
        comp1, comp2, overall_comp = run_algorithm_comparisons(
            output_dir, sarsa_results, q_results, double_q_results
        )
        all_results['sarsa_vs_q'] = comp1
        all_results['q_vs_double_q'] = comp2
        all_results['overall_comparison'] = overall_comp

        # Generate comprehensive report
        report = generate_comprehensive_report(output_dir, all_results)

        print("\n" + "=" * 80)
        print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"Results saved in: {output_dir}")
        print("Files generated:")
        for file in sorted(os.listdir(output_dir)):
            print(f"  - {file}")

    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()

    return all_results, output_dir


if __name__ == "__main__":
    results, output_directory = main()