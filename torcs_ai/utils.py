"""
Utility Functions for TORCS Racing AI

Provides analysis, server management, and helper functions.
"""

import os
import time
import platform
import subprocess
import logging
from typing import Optional
import numpy as np

from .globals import ml_racing_ai, visualizer

logger = logging.getLogger(__name__)

def start_torcs_server() -> bool:
    """
    Automatically start TORCS server for training.

    Returns:
        True if server started successfully, False otherwise
    """
    logger.info("🚀 Starting TORCS server automatically...")

    if platform.system() == 'Windows':
        torcs_path = r'C:\torcs\torcs\wtorcs.exe'
        if os.path.exists(torcs_path):
            try:
                # Start TORCS in background
                subprocess.Popen([torcs_path, '-r', 'quickrace'],
                               creationflags=subprocess.CREATE_NO_WINDOW)
                logger.info("✅ TORCS server started successfully")
                time.sleep(3)  # Give TORCS time to start
                return True
            except Exception as e:
                logger.error(f"❌ Failed to start TORCS: {e}")
                return False
        else:
            logger.error(f"❌ TORCS not found at {torcs_path}")
            logger.error("   Please install TORCS in C:\\torcs\\torcs\\")
            return False
    else:
        # Linux/Mac commands
        try:
            os.system('torcs -nofuel -nodamage -nolaptime &')
            time.sleep(2)
            logger.info("✅ TORCS server started successfully")
            return True
        except Exception as e:
            logger.error("❌ Failed to start TORCS on Linux/Mac")
            return False


def analyze_ml_models() -> None:
    """Analyze ML model performance and generate insights."""
    if not ml_racing_ai.is_trained:
        logger.info("No trained models to analyze")
        return

    try:
        import matplotlib.pyplot as plt

        logger.info("Analyzing ML model performance...")

        # Create test data
        n_test = 1000
        test_features = []
        test_targets = {'steer': [], 'accel': [], 'brake': []}

        for _ in range(n_test):
            speed = np.random.uniform(0, 320)
            angle = np.random.normal(0, 0.3)
            track_pos = np.random.normal(0, 0.5)
            curvature = np.random.uniform(0, 1)

            test_features.append([speed, angle, track_pos, curvature,
                                speed**2, angle**2, abs(track_pos), curvature*speed])

            test_targets['steer'].append(ml_racing_ai.expert_steering(speed, angle, track_pos, curvature))
            test_targets['accel'].append(ml_racing_ai.expert_acceleration(speed, angle, curvature))
            test_targets['brake'].append(ml_racing_ai.expert_braking(speed, angle, curvature))

        X_test = np.array(test_features)

        # Analyze each model
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        for i, action in enumerate(['steer', 'accel', 'brake']):
            if hasattr(ml_racing_ai, 'models') and action in ml_racing_ai.models:
                model = ml_racing_ai.models[action]
                scaler = ml_racing_ai.scalers[action]
                y_true = np.array(test_targets[action])

                # Scale features
                X_test_scaled = scaler.transform(X_test)

                # Predictions
                y_pred = model.predict(X_test_scaled)

                # Feature importance (for sklearn models)
                if hasattr(model, 'feature_importances_'):
                    feature_names = ['speed', 'angle', 'track_pos', 'curvature',
                                   'speed²', 'angle²', '|track_pos|', 'curvature×speed']

                    # Plot feature importance
                    axes[0, i].barh(range(len(feature_names)), model.feature_importances_)
                    axes[0, i].set_yticks(range(len(feature_names)))
                    axes[0, i].set_yticklabels(feature_names)
                    axes[0, i].set_title(f'{action.upper()} Feature Importance')
                    axes[0, i].set_xlabel('Importance')

                # Prediction error distribution
                errors = y_true - y_pred
                axes[1, i].hist(errors, bins=30, alpha=0.7, color=['blue', 'green', 'red'][i])
                axes[1, i].set_title(f'{action.upper()} Prediction Errors')
                axes[1, i].set_xlabel('Error')
                axes[1, i].set_ylabel('Frequency')
                axes[1, i].axvline(x=0, color='black', linestyle='--', alpha=0.5)

                # Print statistics
                mse = np.mean(errors**2)
                mae = np.mean(np.abs(errors))
                logger.info(f"{action.upper()} - MSE: {mse:.4f}, MAE: {mae:.4f}")

        plt.tight_layout()
        plt.savefig('ml_model_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        logger.info("ML model analysis saved as 'ml_model_analysis.png'")

    except ImportError:
        logger.warning("matplotlib not available for ML analysis")
    except Exception as e:
        logger.error(f"Error in ML analysis: {e}")


def generate_racing_insights() -> None:
    """Generate comprehensive insights about racing performance."""
    if len(visualizer.performance_data) < 100:
        logger.info("Not enough data for insights (need at least 100 data points)")
        return

    logger.info("\n🏎️ RACING AI PERFORMANCE INSIGHTS")

    # Analyze speed performance
    speeds = [d['speed'] for d in visualizer.performance_data]
    avg_speed = np.mean(speeds)
    max_speed = np.max(speeds)
    speed_std = np.std(speeds)

    logger.info(f"🏎️ Speed: Avg={avg_speed:.1f}, Max={max_speed:.1f}, Std={speed_std:.1f}")

    # Analyze track position stability
    track_positions = [abs(d['track_pos']) for d in visualizer.performance_data]
    avg_track_error = np.mean(track_positions)
    max_track_error = np.max(track_positions)
    on_track_percentage = sum(1 for pos in track_positions if pos <= 1.0) / len(track_positions) * 100

    logger.info(f"🎯 Track Position: Avg Error={avg_track_error:.3f}, Max Error={max_track_error:.3f}, On Track={on_track_percentage:.1f}%")

    # Analyze reward trends
    rewards = [d['reward'] for d in visualizer.performance_data]
    avg_reward = np.mean(rewards)
    if len(rewards) > 10:
        reward_trend = np.polyfit(range(len(rewards)), rewards, 1)[0]  # Linear trend
        trend_status = "📈 Improving" if reward_trend > 0.001 else "📉 Declining" if reward_trend < -0.001 else "➡️ Stable"
        logger.info(f"💰 Rewards: Avg={avg_reward:.3f}, Trend={reward_trend:+.4f} {trend_status}")
    else:
        logger.info(f"💰 Rewards: Avg={avg_reward:.3f}")

    # Analyze control stability
    steers = [abs(d['steer']) for d in visualizer.performance_data]
    accels = [d['accel'] for d in visualizer.performance_data]
    brakes = [d['brake'] for d in visualizer.performance_data]

    avg_steer = np.mean(steers)
    avg_accel = np.mean(accels)
    avg_brake = np.mean(brakes)

    logger.info(f"🎮 Controls: Steer={avg_steer:.3f}, Accel={avg_accel:.3f}, Brake={avg_brake:.3f}")

    # ML model performance
    if ml_racing_ai.is_trained:
        logger.info("🤖 ML Model Status: ACTIVE")
        logger.info(f"   Training Data: {len(ml_racing_ai.data_collector.experiences)} experiences")
        if hasattr(ml_racing_ai, 'learning_mode') and ml_racing_ai.learning_mode:
            logger.info("   Learning Mode: ENABLED (continuously improving)")
        else:
            logger.info("   Learning Mode: DISABLED (using fixed models)")
    else:
        logger.info("🤖 ML Model Status: INACTIVE")

    # Recommendations
    logger.info("💡 RECOMMENDATIONS:")
    if avg_track_error > 0.3:
        logger.info("   • Improve track position control - car drifting too much")
    if speed_std > 50:
        logger.info("   • Stabilize speed control - too much variation")
    if on_track_percentage < 95:
        logger.info("   • Work on track edge detection and recovery")
    if avg_reward < 0.5:
        logger.info("   • Optimize reward function or driving strategy")
    if ml_racing_ai.is_trained and len(ml_racing_ai.data_collector.experiences) < 1000:
        logger.info("   • Collect more training data for better ML performance")

    # Generate performance report
    report = visualizer.generate_performance_report()
    logger.info(f"\n📊 DETAILED PERFORMANCE REPORT:\n{report}")


def analyze_track_curvature(sensor_data: dict) -> tuple:
    """Analyze track curvature from sensor data."""
    track = sensor_data.get('track', [])
    if len(track) < 5:
        return 0.0, []

    # Calculate curvature using finite differences
    curvatures = []
    for i in range(2, len(track) - 2):
        # Second derivative approximation
        curvature = track[i-2] - 4*track[i-1] + 6*track[i] - 4*track[i+1] + track[i+2]
        curvature /= 12  # Normalize
        curvatures.append(abs(curvature))

    avg_curvature = np.mean(curvatures) if curvatures else 0.0
    return avg_curvature, curvatures


def detect_racing_scenarios(sensor_data: dict) -> dict:
    """Detect current racing scenario for adaptive behavior."""
    scenarios = {
        'emergency': False,
        'wall_proximity': False,
        'spin_recovery': False,
        'high_speed_corner': False,
        'hairpin_corner': False,
        'straight_high_speed': False,
        'crowded_track': False,
        'low_fuel': False,
        'damage_critical': False
    }

    speed = sensor_data.get('speedX', 0)
    angle = abs(sensor_data.get('angle', 0))
    track_pos = abs(sensor_data.get('trackPos', 0))
    damage = sensor_data.get('damage', 0)
    fuel = sensor_data.get('fuel', 100)
    curvature, _ = analyze_track_curvature(sensor_data)

    # Emergency situations
    scenarios['emergency'] = damage > 5000 or track_pos >= 1.5 or angle > 1.0
    scenarios['wall_proximity'] = track_pos > 0.8
    scenarios['spin_recovery'] = angle > 0.8
    scenarios['damage_critical'] = damage > 3000
    scenarios['low_fuel'] = fuel < 20

    # Track conditions
    scenarios['high_speed_corner'] = speed > 150 and curvature > 0.5
    scenarios['hairpin_corner'] = curvature > 1.0
    scenarios['straight_high_speed'] = speed > 250 and curvature < 0.2
    scenarios['crowded_track'] = any(op > 50 for op in sensor_data.get('opponents', []))

    return scenarios


def calculate_adaptive_exploration(scenarios: dict) -> float:
    """Calculate exploration rate based on current scenarios."""
    base_rate = 0.02  # Conservative base rate

    # Reduce exploration in dangerous situations
    if scenarios['emergency'] or scenarios['wall_proximity'] or scenarios['spin_recovery']:
        return 0.001  # Minimal exploration in emergencies

    # Increase exploration in safe learning situations
    if scenarios['straight_high_speed'] and not scenarios['crowded_track']:
        return 0.05  # More exploration on safe straights

    # Moderate exploration in normal racing
    if scenarios['high_speed_corner'] or scenarios['crowded_track']:
        return 0.01  # Reduced exploration in complex situations

    return base_rate


def get_scenario_noise_scale(scenarios: dict, action: str) -> float:
    """Get appropriate noise scale for different scenarios and actions."""
    if action == 'steer':
        base_noise = 0.02

        # Reduce steering noise in dangerous situations
        if scenarios['emergency'] or scenarios['wall_proximity']:
            return 0.005

        # Increase precision in corners
        if scenarios['hairpin_corner'] or scenarios['high_speed_corner']:
            return 0.01

        return base_noise

    elif action == 'accel':
        base_noise = 0.01

        # Conservative acceleration noise in crowded situations
        if scenarios['crowded_track']:
            return 0.005

        return base_noise

    else:  # brake
        return 0.005  # Always conservative braking noise


def calculate_adaptive_smoothing(scenarios: dict) -> dict:
    """Calculate smoothing factors based on scenarios."""
    base_steer_smooth = 0.25
    base_accel_smooth = 0.15
    base_brake_smooth = 0.3

    # Increase smoothing in unstable situations
    if scenarios['emergency'] or scenarios['spin_recovery']:
        return {
            'steer': min(0.5, base_steer_smooth * 2),
            'accel': min(0.4, base_accel_smooth * 2),
            'brake': min(0.6, base_brake_smooth * 1.5)
        }

    # Reduce smoothing for precision in corners
    if scenarios['hairpin_corner'] or scenarios.get('chicane_sequence', False):
        return {
            'steer': max(0.1, base_steer_smooth * 0.7),
            'accel': max(0.05, base_accel_smooth * 0.8),
            'brake': max(0.1, base_brake_smooth * 0.8)
        }

    return {
        'steer': base_steer_smooth,
        'accel': base_accel_smooth,
        'brake': base_brake_smooth
    }


def apply_scenario_adaptations(sensor_data: dict, actions: dict, scenarios: dict) -> None:
    """Apply scenario-based adaptations to driving actions."""
    speed = sensor_data.get('speedX', 0)
    track_pos = sensor_data.get('trackPos', 0)

    # Emergency adaptations
    if scenarios['emergency']:
        actions['brake'] = max(actions['brake'], 0.5)  # Emergency braking
        actions['accel'] = 0.0  # No acceleration in emergency

    # Wall proximity adaptations
    if scenarios['wall_proximity']:
        # Steer away from wall
        wall_direction = 1 if track_pos > 0 else -1
        actions['steer'] += wall_direction * 0.2
        actions['accel'] *= 0.7  # Reduce speed near walls

    # High speed corner adaptations
    if scenarios['high_speed_corner']:
        actions['accel'] *= 0.8  # Conservative acceleration in corners
        actions['brake'] = max(actions['brake'], 0.1)  # Light braking anticipation


def calculate_competitive_strategy(sensor_data: dict, scenarios: dict) -> dict:
    """Calculate competitive racing strategy."""
    strategy = {
        'aggressive': False,
        'defensive': False,
        'overtaking': False,
        'fuel_conservation': False
    }

    opponents = sensor_data.get('opponents', [])
    speed = sensor_data.get('speedX', 0)
    fuel = sensor_data.get('fuel', 100)

    # Overtaking opportunities
    if any(op < 10 for op in opponents[:4]):  # Opponents very close ahead
        strategy['overtaking'] = True
        strategy['aggressive'] = True

    # Defensive driving
    if any(op < 20 for op in opponents):  # Opponents nearby
        strategy['defensive'] = True

    # Fuel conservation
    if fuel < 30:
        strategy['fuel_conservation'] = True

    return strategy


def apply_competitive_strategy(actions: dict, strategy: dict, sensor_data: dict) -> None:
    """Apply competitive strategy modifications."""
    if strategy['aggressive']:
        actions['accel'] *= 1.2
        actions['brake'] *= 0.8

    if strategy['fuel_conservation']:
        actions['accel'] *= 0.9
        actions['steer'] *= 0.95  # Smoother steering

    if strategy['overtaking']:
        # Adjust steering for overtaking maneuvers
        actions['steer'] *= 1.1


def shift_gears(sensor_data: dict) -> int:
    """Optimized gear shifting logic."""
    speed = sensor_data.get('speedX', 0)

    if speed > 140:
        return 3
    elif speed > 100:
        return 2
    elif speed > 60:
        return 1
    return 0


def calculate_steering(sensor_data: dict) -> float:
    """Calculate steering angle based on sensor data."""
    angle = sensor_data.get('angle', 0)
    track_pos = sensor_data.get('trackPos', 0)
    curvature, _ = analyze_track_curvature(sensor_data)

    steer = angle * 25 / np.pi - track_pos * 0.25

    # Adjust for curvature
    if curvature > 0.3:
        steer *= 1.2

    return np.clip(steer, -1, 1)


def calculate_throttle(sensor_data: dict, actions: dict) -> float:
    """Calculate throttle based on sensor data."""
    speed = sensor_data.get('speedX', 0)
    angle = abs(sensor_data.get('angle', 0))
    curvature, _ = analyze_track_curvature(sensor_data)

    target_speed = 280 * (1 - curvature * 0.4)
    speed_error = target_speed - speed

    if abs(angle) > 0.3:
        return 0.3  # Conservative in corners
    elif speed_error > 20:
        return 1.0
    elif speed_error > 0:
        return 0.7
    else:
        return 0.2


def apply_brakes(sensor_data: dict) -> float:
    """Calculate braking based on sensor data."""
    speed = sensor_data.get('speedX', 0)
    angle = abs(sensor_data.get('angle', 0))
    curvature, _ = analyze_track_curvature(sensor_data)

    if abs(angle) > 0.4 or (speed > 250 and curvature > 0.4):
        return min(0.8, abs(angle) * 2)
    return 0.0


def update_state_and_reward(sensor_data: dict, actions: dict, scenarios: dict,
                          prev_damage: float, prev_fuel: float) -> tuple:
    """Update internal state and calculate reward."""
    current_damage = sensor_data.get('damage', 0)
    current_fuel = sensor_data.get('fuel', 100)

    reward = ml_racing_ai.calculate_reward(sensor_data, actions)

    # Additional scenario-based rewards/penalties
    if scenarios['emergency']:
        reward -= 1.0  # Penalty for emergencies
    if scenarios['wall_proximity']:
        reward -= 0.5  # Penalty for being near walls

    return reward, current_damage, current_fuel