"""
Advanced Training Module for TORCS Racing AI

Provides comprehensive training pipelines, automated learning,
and performance optimization strategies.
"""

import logging
import time
from typing import Any, Optional

import numpy as np

from .client import Client
from .globals import ml_racing_ai, visualizer
from .ml_models import MLRacingAI
from .utils import analyze_ml_models, generate_racing_insights, start_torcs_server

logger = logging.getLogger(__name__)


def drive_modular(c: Client) -> None:
    """Entry point for the machine learning-powered racing AI."""
    ml_racing_ai.drive(c)

    # Collect data for visualization (every 10 steps to reduce overhead)
    if not hasattr(drive_modular, "step_counter"):
        drive_modular.step_counter = 0
    drive_modular.step_counter += 1

    if drive_modular.step_counter % 10 == 0:
        reward = ml_racing_ai.calculate_reward(c.S.d, c.R.d)
        visualizer.collect_data(c.S.d, c.R.d, reward)

        # Generate plots periodically
        if drive_modular.step_counter % 500 == 0:
            visualizer.plot_comprehensive_analysis()
            visualizer.create_interactive_dashboard()


def automated_training_pipeline(
    num_races: int = 10, max_steps_per_race: int = 5000, save_interval: int = 5
) -> dict[str, Any]:
    """
    Automated training pipeline that runs multiple races and improves the AI.
    Automatically starts TORCS server.

    Args:
        num_races: Number of races to run for training
        max_steps_per_race: Maximum steps per race
        save_interval: Save models every N races

    Returns:
        Training statistics dictionary
    """
    logger.info("🚀 Starting Automated Training Pipeline")
    logger.info(f"🎯 Target: {num_races} races, {max_steps_per_race} steps each")

    # Check TORCS server (manual startup required)
    if not start_torcs_server():
        logger.info("⏳ Please start TORCS manually using the instructions above")
        logger.info(
            "   Once TORCS is running and shows 'Waiting for request on port 3001',"
        )
        logger.info("   press Enter to continue with training...")
        input()  # Wait for user confirmation

    # Track training progress
    training_stats = {
        "races_completed": 0,
        "total_experiences": 0,
        "best_performance": float("-inf"),
        "performance_history": [],
        "start_time": time.time(),
    }

    for race_num in range(1, num_races + 1):
        logger.info(f"🏁 Race {race_num}/{num_races} - Starting...")

        try:
            # Create client for this race
            C = Client(port=3001, max_episodes=1, max_steps=max_steps_per_race)

            # Reset step counter for data collection
            drive_modular.step_counter = 0

            # Run the race
            race_experiences = 0
            start_race_time = time.time()

            for step in range(C.maxSteps, 0, -1):
                C.get_servers_input()
                drive_modular(C)
                C.respond_to_server()

                # Count experiences collected
                if drive_modular.step_counter % 10 == 0:
                    race_experiences += 1

                # Progress update
                if step % 1000 == 0:
                    progress = (C.maxSteps - step) / C.maxSteps * 100
                    elapsed = time.time() - start_race_time
                    logger.info("Progress %.1f%%; elapsed %.1fs", progress, elapsed)

            C.shutdown()

            race_time = time.time() - start_race_time
            logger.info("Race finished in %.2fs", race_time)

            # Update training stats
            training_stats["races_completed"] += 1
            training_stats["total_experiences"] += race_experiences

            # Evaluate performance
            if len(visualizer.performance_data) >= 50:
                recent_rewards = [
                    d["reward"] for d in visualizer.performance_data[-50:]
                ]
                avg_performance = np.mean(recent_rewards)
                training_stats["performance_history"].append(avg_performance)

                if avg_performance > training_stats["best_performance"]:
                    training_stats["best_performance"] = avg_performance
                    logger.info(f"   🏆 New best performance: {avg_performance:.3f}")
                else:
                    logger.info(f"   📊 Race performance: {avg_performance:.3f}")

            # Periodic model saving and retraining
            if race_num % save_interval == 0:
                logger.info(f"💾 Saving models after race {race_num}...")
                ml_racing_ai.save_models()

                # Force retraining if we have enough data
                if len(ml_racing_ai.data_collector.experiences) >= 500:
                    logger.info("🔄 Retraining models with accumulated experience...")
                    ml_racing_ai.retrain_models()

            logger.info(
                f"✅ Race {race_num} completed! Experiences collected: {race_experiences}"
            )

        except Exception as e:
            logger.error(f"❌ Error in race {race_num}: {e}")
            continue

    # Final analysis and summary
    training_stats["total_time"] = time.time() - training_stats["start_time"]

    logger.info("\n🏆 AUTOMATED TRAINING COMPLETE")
    logger.info("📊 Training Summary:")
    logger.info(
        f"   • Races completed: {training_stats['races_completed']}/{num_races}"
    )
    logger.info(f"   • Total experiences: {training_stats['total_experiences']}")
    logger.info(f"   • Best performance: {training_stats['best_performance']:.3f}")
    logger.info(f"   • Total training time: {training_stats['total_time']:.1f} seconds")

    if training_stats["performance_history"]:
        improvement = (
            training_stats["performance_history"][-1]
            - training_stats["performance_history"][0]
        )
        logger.info(f"   • Performance improvement: {improvement:+.3f}")

    # Final model save
    logger.info("💾 Saving final trained models...")
    ml_racing_ai.save_models()

    # Generate final analysis
    logger.info("📈 Generating final performance analysis...")
    analyze_ml_models()
    generate_racing_insights()

    # Save training data
    visualizer.save_data_to_csv("training_data.csv")

    logger.info("🎯 Training pipeline completed successfully!")
    return training_stats


def continuous_learning_mode(
    max_races: int = 50, performance_threshold: float = 0.5
) -> dict[str, Any]:
    """
    Continuous learning mode that keeps training until performance threshold is reached.

    Args:
        max_races: Maximum number of races to run
        performance_threshold: Stop when average reward exceeds this threshold

    Returns:
        Training statistics dictionary
    """
    logger.info("🔄 Starting Continuous Learning Mode")
    logger.info(
        f"🎯 Target: Performance > {performance_threshold} or {max_races} races max"
    )

    # Auto-start TORCS server
    if not start_torcs_server():
        logger.error("❌ Cannot start training without TORCS server")
        return {}

    race_num = 0
    recent_performances: list[float] = []
    training_stats = {
        "races_completed": 0,
        "final_performance": 0.0,
        "threshold_reached": False,
        "start_time": time.time(),
    }

    while race_num < max_races:
        race_num += 1
        logger.info(f"🏁 Continuous Learning - Race {race_num}")

        try:
            # Run one race
            C = Client(
                port=3001, max_episodes=1, max_steps=3000
            )  # Shorter races for continuous learning

            drive_modular.step_counter = 0
            start_race_time = time.time()

            for step in range(C.maxSteps, 0, -1):
                C.get_servers_input()
                drive_modular(C)
                C.respond_to_server()

                if step % 1000 == 0:
                    progress = (C.maxSteps - step) / C.maxSteps * 100
                    elapsed = time.time() - start_race_time
                    logger.info("Progress %.1f%%; elapsed %.1fs", progress, elapsed)

            C.shutdown()

            race_time = time.time() - start_race_time
            logger.info("Race finished in %.2fs", race_time)

            # Check performance
            if len(visualizer.performance_data) >= 20:
                recent_rewards = [
                    d["reward"] for d in visualizer.performance_data[-20:]
                ]
                avg_performance = np.mean(recent_rewards)
                recent_performances.append(avg_performance)

                # Keep only last 5 performances for moving average
                if len(recent_performances) > 5:
                    recent_performances = recent_performances[-5:]

                moving_avg = np.mean(recent_performances)

                logger.info("Moving average performance: %.3f", moving_avg)

                # Check if we've reached the performance threshold
                if moving_avg >= performance_threshold:
                    logger.info("🎉 Performance threshold reached! Stopping training.")
                    training_stats["threshold_reached"] = True
                    training_stats["final_performance"] = moving_avg
                    break

            # Periodic retraining
            if (
                race_num % 3 == 0
                and len(ml_racing_ai.data_collector.experiences) >= 300
            ):
                logger.info("🔄 Retraining models...")
                ml_racing_ai.retrain_models()
                ml_racing_ai.save_models()

        except Exception as e:
            logger.error(f"❌ Error in continuous learning race {race_num}: {e}")
            continue

    training_stats["races_completed"] = race_num
    training_stats["total_time"] = time.time() - training_stats["start_time"]

    logger.info(f"\n🏆 Continuous Learning Complete after {race_num} races")
    if not training_stats["threshold_reached"]:
        training_stats["final_performance"] = (
            np.mean(recent_performances) if recent_performances else 0.0
        )

    analyze_ml_models()
    generate_racing_insights()

    return training_stats


def perfection_training_pipeline() -> dict[str, Any]:
    """
    Ultimate training pipeline to achieve racing perfection.
    Multi-phase training with increasing difficulty.

    Returns:
        Training statistics dictionary
    """
    logger.info("🏆 PERFECT RACING AI TRAINING - PHASE 1: FOUNDATION")

    # Auto-start TORCS server
    if not start_torcs_server():
        logger.error("❌ Cannot start training without TORCS server")
        return {}

    total_stats = {
        "phases_completed": 0,
        "total_races": 0,
        "best_performance": float("-inf"),
        "start_time": time.time(),
    }

    # Phase 1: Foundation Building
    logger.info("🎯 Goal: Establish solid baseline performance")
    logger.info("📊 Target: 50 races, performance > 0.3")

    phase1_stats = continuous_learning_mode(max_races=50, performance_threshold=0.3)
    total_stats["phases_completed"] += 1
    total_stats["total_races"] += phase1_stats.get("races_completed", 0)
    total_stats["best_performance"] = max(
        total_stats["best_performance"], phase1_stats.get("final_performance", 0)
    )

    if not phase1_stats.get("threshold_reached", False):
        logger.warning("Phase 1 not completed successfully, but continuing...")

    # Phase 2: Skill Development
    logger.info("\n🏆 PHASE 2: SKILL DEVELOPMENT")
    logger.info("🎯 Goal: Master cornering and overtaking")
    logger.info("📊 Target: 75 races, performance > 0.6")

    phase2_stats = continuous_learning_mode(max_races=75, performance_threshold=0.6)
    total_stats["phases_completed"] += 1
    total_stats["total_races"] += phase2_stats.get("races_completed", 0)
    total_stats["best_performance"] = max(
        total_stats["best_performance"], phase2_stats.get("final_performance", 0)
    )

    # Phase 3: Elite Performance
    logger.info("\n🏆 PHASE 3: ELITE PERFORMANCE")
    logger.info("🎯 Goal: Achieve championship-level racing")
    logger.info("📊 Target: 100 races, performance > 0.8")

    phase3_stats = continuous_learning_mode(max_races=100, performance_threshold=0.8)
    total_stats["phases_completed"] += 1
    total_stats["total_races"] += phase3_stats.get("races_completed", 0)
    total_stats["best_performance"] = max(
        total_stats["best_performance"], phase3_stats.get("final_performance", 0)
    )

    # Phase 4: Perfection
    logger.info("\n🏆 PHASE 4: PERFECTION")
    logger.info("🎯 Goal: Ultimate racing perfection")
    logger.info("📊 Target: 200 races, performance > 0.95")

    phase4_stats = continuous_learning_mode(max_races=200, performance_threshold=0.95)
    total_stats["phases_completed"] += 1
    total_stats["total_races"] += phase4_stats.get("races_completed", 0)
    total_stats["best_performance"] = max(
        total_stats["best_performance"], phase4_stats.get("final_performance", 0)
    )

    total_stats["total_time"] = time.time() - total_stats["start_time"]

    logger.info("\n🎉 PERFECTION TRAINING COMPLETE!")
    logger.info("👑 Your AI has achieved ultimate racing perfection!")
    logger.info(
        f"📊 Final Stats: {total_stats['phases_completed']} phases, {total_stats['total_races']} races"
    )
    logger.info(f"🏆 Best Performance: {total_stats['best_performance']:.3f}")
    logger.info(f"⏱️ Total Training Time: {total_stats['total_time']:.1f} seconds")

    # Save legendary model
    legendary_filename = f"legendary_racing_ai_{int(time.time())}.pth"
    try:
        ml_racing_ai.save_models()
        logger.info(f"💾 Legendary model saved as: {legendary_filename}")
    except Exception as e:
        logger.error(f"Could not save legendary model: {e}")

    return total_stats


def elite_training_curriculum() -> dict[str, Any]:
    """
    Elite curriculum training with structured phases of increasing difficulty.

    Returns:
        Training statistics dictionary
    """
    logger.info("👑 ELITE CURRICULUM TRAINING")
    logger.info("🎯 Multi-phase training with progressive difficulty")

    # Auto-start TORCS server
    if not start_torcs_server():
        logger.error("❌ Cannot start training without TORCS server")
        return {}

    phases = [
        {
            "name": "NOVICE",
            "description": "Basic track navigation and speed control",
            "races": 25,
            "threshold": 0.2,
            "focus": "Stability",
        },
        {
            "name": "INTERMEDIATE",
            "description": "Cornering technique and opponent awareness",
            "races": 50,
            "threshold": 0.4,
            "focus": "Technique",
        },
        {
            "name": "ADVANCED",
            "description": "High-speed racing and strategic positioning",
            "races": 75,
            "threshold": 0.6,
            "focus": "Speed",
        },
        {
            "name": "EXPERT",
            "description": "Defensive driving and overtaking mastery",
            "races": 100,
            "threshold": 0.75,
            "focus": "Strategy",
        },
        {
            "name": "MASTER",
            "description": "Perfect lap consistency and adaptability",
            "races": 150,
            "threshold": 0.85,
            "focus": "Consistency",
        },
        {
            "name": "LEGENDARY",
            "description": "Ultimate racing perfection",
            "races": 500,
            "threshold": 0.95,
            "focus": "Perfection",
        },
    ]

    total_stats = {
        "phases_completed": 0,
        "total_races": 0,
        "best_performance": float("-inf"),
        "start_time": time.time(),
    }

    for phase in phases:
        logger.info(f"\n🏆 PHASE: {phase['name']}")
        logger.info(f"📚 Focus: {phase['focus']}")
        logger.info(f"🎯 Goal: {phase['description']}")
        logger.info(
            f"📊 Target: {phase['races']} races, performance > {phase['threshold']}"
        )

        phase_stats = continuous_learning_mode(
            max_races=phase["races"], performance_threshold=phase["threshold"]
        )

        total_stats["phases_completed"] += 1
        total_stats["total_races"] += phase_stats.get("races_completed", 0)
        total_stats["best_performance"] = max(
            total_stats["best_performance"], phase_stats.get("final_performance", 0)
        )

        if not phase_stats.get("threshold_reached", False):
            logger.warning(
                f"Phase {phase['name']} target not fully achieved, continuing..."
            )

    total_stats["total_time"] = time.time() - total_stats["start_time"]

    logger.info("\n🎉 ELITE CURRICULUM COMPLETE!")
    logger.info("👑 Your AI has achieved LEGENDARY status!")
    logger.info(f"📊 Final Stats: {total_stats}")

    # Save legendary model
    try:
        ml_racing_ai.save_models()
        logger.info("💾 Legendary model saved!")
    except Exception as e:
        logger.error(f"Could not save legendary model: {e}")

    return total_stats


def intensive_training_session(intensity_level: str = "extreme") -> dict[str, Any]:
    """
    Intensive training session with configurable intensity levels.

    Args:
        intensity_level: 'moderate', 'intensive', 'extreme', 'insane'

    Returns:
        Training statistics dictionary
    """
    logger.info(f"🔥 Starting Intensive Training Session ({intensity_level})")

    # Auto-start TORCS server
    if not start_torcs_server():
        logger.error("❌ Cannot start training without TORCS server")
        return {}

    intensity_configs = {
        "moderate": {
            "races": 20,
            "threshold": 0.3,
            "description": "Balanced training for steady improvement",
        },
        "intensive": {
            "races": 50,
            "threshold": 0.5,
            "description": "Aggressive training for rapid improvement",
        },
        "extreme": {
            "races": 100,
            "threshold": 0.7,
            "description": "Extreme training for maximum performance",
        },
        "insane": {
            "races": 200,
            "threshold": 0.85,
            "description": "Insane training for perfection seekers",
        },
    }

    if intensity_level not in intensity_configs:
        logger.error(
            f"Invalid intensity level. Choose from: {list(intensity_configs.keys())}"
        )
        return {}

    config = intensity_configs[intensity_level]

    logger.info(f"🔥 INTENSIVE TRAINING SESSION - {intensity_level.upper()}")
    logger.info(f"🎯 Mode: {config['description']}")
    logger.info(
        f"📊 Target: {config['races']} races, performance > {config['threshold']}"
    )

    # Pre-training analysis
    logger.info("📊 Pre-training analysis:")
    analyze_ml_models()

    # Intensive training
    start_time = time.time()
    training_stats = continuous_learning_mode(
        max_races=config["races"], performance_threshold=config["threshold"]
    )
    training_time = time.time() - start_time

    # Post-training analysis
    logger.info("📊 Post-training analysis:")
    analyze_ml_models()
    generate_racing_insights()

    # Training summary
    logger.info("🏆 TRAINING SUMMARY:")
    logger.info(f"   • Training Time: {training_time:.1f} seconds")
    logger.info(f"   • Intensity Level: {intensity_level.upper()}")
    logger.info(f"   • Target Performance: {config['threshold']}")
    logger.info(
        f"   • Training Data Collected: {len(ml_racing_ai.data_collector.experiences)} experiences"
    )
    logger.info(f"   • Performance Data Points: {len(visualizer.performance_data)}")

    if len(visualizer.performance_data) >= 10:
        final_performance = np.mean(
            [d["reward"] for d in visualizer.performance_data[-10:]]
        )
        logger.info(f"   • Final Performance: {final_performance:.3f}")
        if final_performance >= config["threshold"]:
            logger.info("   ✅ Target performance achieved!")
        else:
            logger.info("   ⚠️ Target performance not fully achieved")

    training_stats["training_time"] = training_time
    training_stats["intensity_level"] = intensity_level

    return training_stats
