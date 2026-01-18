#!/usr/bin/env python3
"""
TORCS ML Racing AI - Demonstration Script
Shows the ML capabilities without requiring TORCS server
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from torcs_jm_par import ml_racing_ai, visualizer, analyze_ml_models, generate_racing_insights
import numpy as np

def demo_ml_predictions():
    """Demonstrate ML model predictions with sample data."""
    print("🚗 TORCS ML Racing AI - Prediction Demo")
    print("="*50)

    # Sample racing scenarios
    scenarios = [
        {
            'name': 'Straight section',
            'data': {
                'speed': 80, 'angle': 0.0, 'track_pos': 0.1,
                'track': [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
                'opponents': [200]*36,
                'rpm': 8000, 'gear': 3
            }
        },
        {
            'name': 'Sharp corner approaching',
            'data': {
                'speed': 60, 'angle': 0.3, 'track_pos': -0.2,
                'track': [5, 3, 2, 1, 0.5, 2, 5, 8, 10, 12, 15, 18, 20, 22, 25, 28, 30, 35, 40],
                'opponents': [200]*36,
                'rpm': 6000, 'gear': 2
            }
        },
        {
            'name': 'Opponent ahead',
            'data': {
                'speed': 70, 'angle': -0.1, 'track_pos': 0.0,
                'track': [15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15],
                'opponents': [50, 45, 40] + [200]*33,  # Opponents close ahead
                'rpm': 7000, 'gear': 3
            }
        }
    ]

    for scenario in scenarios:
        print(f"\n📍 {scenario['name']}:")
        predictions = ml_racing_ai.predict_action(scenario['data'])

        if predictions:
            steer_pred = predictions['steer']
            accel_pred = predictions['accel']
            brake_pred = predictions['brake']
        else:
            print("   No predictions available")
            continue

        print(f"   Steering: {steer_pred:.3f}")
        print(f"   Acceleration: {accel_pred:.3f}")
        print(f"   Braking: {brake_pred:.3f}")
        # Calculate reward for this scenario
        reward = ml_racing_ai.calculate_reward(scenario['data'], {
            'steer': steer_pred,
            'accel': accel_pred,
            'brake': brake_pred
        })
        print(f"   Predicted Reward: {reward:.3f}")
def demo_data_collection():
    """Demonstrate data collection capabilities."""
    print("\n📊 Data Collection Demo")
    print("="*30)

    # Simulate collecting some racing data
    for i in range(5):
        sample_data = {
            'speed': np.random.uniform(40, 120),
            'angle': np.random.uniform(-0.5, 0.5),
            'track_pos': np.random.uniform(-1, 1),
            'track': np.random.uniform(5, 50, 19).tolist(),
            'opponents': np.random.uniform(50, 200, 36).tolist(),
            'rpm': np.random.uniform(3000, 9000),
            'gear': np.random.randint(1, 6),
            'steer': np.random.uniform(-0.8, 0.8),
            'accel': np.random.uniform(0, 1),
            'brake': np.random.uniform(0, 1),
            'reward': np.random.uniform(-1, 1)
        }
        actions = {
            'steer': sample_data['steer'],
            'accel': sample_data['accel'],
            'brake': sample_data['brake']
        }
        visualizer.collect_data(sample_data, actions, sample_data['reward'])

    print(f"Collected {len(visualizer.performance_data)} data points")
    print("Data includes: speed, angle, track position, sensor readings, controls, rewards")

def main():
    print("🤖 TORCS Machine Learning Racing AI Demo")
    print("This demonstrates the ML capabilities without running TORCS")
    print("="*60)

    # Check if models are loaded
    if not ml_racing_ai.is_trained:
        print("❌ ML models not loaded. Please run torcs_jm_par.py first to train models.")
        return

    # Run demonstrations
    demo_ml_predictions()
    demo_data_collection()

    print("\n" + "="*60)
    print("📈 Running full analysis...")
    analyze_ml_models()

    if len(visualizer.performance_data) >= 100:
        generate_racing_insights()
    else:
        print("💡 Need more data for racing insights (currently have", len(visualizer.performance_data), "points)")

    print("\n✅ Demo complete! The ML racing AI is ready for TORCS.")
    print("Run 'python torcs_jm_par.py' to start racing with ML-powered AI.")

if __name__ == "__main__":
    main()