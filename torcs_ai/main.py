#!/usr/bin/env python3
"""
TORCS Racing AI - Main Entry Point

A sophisticated machine learning-based racing AI for TORCS (The Open Racing Car Simulator).
Features advanced neural networks, automated training, real-time visualization, and continuous learning.
"""

import logging
import sys
import time

logger = logging.getLogger(__name__)


def main():
    """Main entry point for TORCS Racing AI."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == 'analyze':
            from torcs_ai.utils import analyze_ml_models, generate_racing_insights

            # Run analysis mode
            logger.info("🔍 Running ML model analysis...")
            analyze_ml_models()
            generate_racing_insights()

        elif command == 'train':
            from torcs_ai.training import automated_training_pipeline

            # Automated training pipeline
            num_races = int(sys.argv[2]) if len(sys.argv) > 2 else 10
            logger.info(f"🚀 Starting automated training pipeline with {num_races} races...")
            stats = automated_training_pipeline(num_races=num_races)
            logger.info(f"✅ Training completed. Stats: {stats}")

        elif command == 'continuous':
            from torcs_ai.training import continuous_learning_mode

            # Continuous learning mode
            max_races = int(sys.argv[2]) if len(sys.argv) > 2 else 50
            threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
            logger.info(f"🔄 Starting continuous learning (max {max_races} races, threshold {threshold})...")
            stats = continuous_learning_mode(max_races=max_races, performance_threshold=threshold)
            logger.info(f"✅ Continuous learning completed. Stats: {stats}")

        elif command == 'perfection':
            from torcs_ai.training import perfection_training_pipeline

            # Ultimate perfection training
            logger.info("🏆 Starting perfection training pipeline...")
            stats = perfection_training_pipeline()
            logger.info(f"✅ Perfection training completed. Stats: {stats}")

        elif command == 'elite':
            from torcs_ai.training import elite_training_curriculum

            # Elite curriculum training
            logger.info("👑 Starting elite curriculum training...")
            stats = elite_training_curriculum()
            logger.info(f"✅ Elite training completed. Stats: {stats}")

        elif command == 'intensive':
            from torcs_ai.training import intensive_training_session

            # Intensive training session
            intensity = sys.argv[2] if len(sys.argv) > 2 else 'extreme'
            logger.info(f"🔥 Starting intensive training ({intensity})...")
            stats = intensive_training_session(intensity_level=intensity)
            logger.info(f"✅ Intensive training completed. Stats: {stats}")

        elif command == 'demo':
            from torcs_ai.globals import ml_racing_ai, visualizer

            # Demo mode - show training capabilities without TORCS
            print("🎯 TORCS ML Racing AI - Advanced Training Demo")
            print("="*60)
            print("🚀 Available Training Modes:")
            print("   1. analyze          - Analyze current ML models")
            print("   2. train N          - Run automated training pipeline (N races)")
            print("   3. continuous N T   - Continuous learning until performance T")
            print("   4. perfection       - Ultimate perfection training")
            print("   5. elite           - Elite curriculum training")
            print("   6. intensive L     - Intensive training (L=moderate/extreme/insane)")
            print("   7. demo            - Show this demo")
            print("   8. help            - Show usage instructions")
            print()
            print("🤖 Advanced Features:")
            print("   • Deep Neural Networks (PyTorch)")
            print("   • Deep Q-Learning for decision making")
            print("   • Real-time visualization and analytics")
            print("   • Adaptive exploration and learning")
            print("   • Scenario-aware driving strategies")
            print("   • Comprehensive performance tracking")
            print()
            print("📊 Current Status:")
            print(f"   • ML Models: {'LOADED' if ml_racing_ai.is_trained else 'NOT TRAINED'}")
            print(f"   • Training Data: {len(ml_racing_ai.data_collector.experiences)} experiences")
            print(f"   • Performance Data: {len(visualizer.performance_data)} points")
            print()
            print("💡 To start automated training:")
            print("   1. Start TORCS server")
            print("   2. Run: python -m torcs_ai.main train 5")
            print("   3. Watch the AI learn and improve automatically!")
            print("="*60)

        elif command == 'help':
            print("🏎️  TORCS ML Racing AI - Advanced Usage Guide")
            print("="*55)
            print("python -m torcs_ai.main              # Run single race")
            print("python -m torcs_ai.main analyze      # Analyze current models")
            print("python -m torcs_ai.main train [N]    # Automated training (N races)")
            print("python -m torcs_ai.main continuous [N] [T]  # Continuous learning")
            print("                                      # N=max races, T=performance threshold")
            print("python -m torcs_ai.main perfection   # Ultimate perfection training")
            print("python -m torcs_ai.main elite        # Elite curriculum training")
            print("python -m torcs_ai.main intensive [L] # Intensive training (L=moderate/hard/extreme/insane)")
            print("python -m torcs_ai.main demo         # Show training capabilities")
            print("python -m torcs_ai.main help         # Show this help")
            print()
            print("🎯 Advanced Features:")
            print("   • Neural Network Models with PyTorch")
            print("   • Deep Reinforcement Learning")
            print("   • Real-time Performance Visualization")
            print("   • Adaptive Learning Strategies")
            print("   • Comprehensive Analytics and Insights")
            print("   • Automated Server Management")
            print("="*55)

        else:
            print(f"❌ Unknown command: {command}")
            print("Use 'python -m torcs_ai.main demo' for available options.")

    else:
        from torcs_ai.client import Client
        from torcs_ai.globals import ml_racing_ai, visualizer
        from torcs_ai.training import drive_modular
        from torcs_ai.utils import analyze_ml_models, generate_racing_insights

        # Run racing mode (default)
        logger.info("🏎️ Starting Advanced Machine Learning Racing AI...")
        logger.info("🤖 Neural Networks: LOADED")
        logger.info("📊 Real-time Analytics: ENABLED")
        logger.info("🎯 Target: Ultimate racing performance with continuous learning")

        try:
            C = Client(port=3001)
            race_start_time = time.time()

            for step in range(C.maxSteps, 0, -1):
                C.get_servers_input()
                drive_modular(C)
                C.respond_to_server()

                # Periodic analysis
                if step % 1000 == 0:
                    progress = (C.maxSteps - step) / C.maxSteps * 100
                    elapsed = time.time() - race_start_time
                    logger.info("Progress %.1f%%; elapsed %.1fs", progress, elapsed)

            C.shutdown()

            race_time = time.time() - race_start_time
            logger.info("Race finished in %.2fs", race_time)

            # Final analysis
            logger.info("🏁 RACE COMPLETE - Generating final analysis...")
            analyze_ml_models()
            generate_racing_insights()

            # Save race data
            visualizer.save_data_to_csv('race_data.csv')
            visualizer.plot_comprehensive_analysis('final_race_analysis.png')
            visualizer.create_interactive_dashboard('final_race_dashboard.html')

            logger.info("✅ Race completed successfully!")

        except KeyboardInterrupt:
            logger.info("🛑 Race interrupted by user")
        except Exception as e:
            logger.error(f"❌ Error during race: {e}")
            raise


if __name__ == "__main__":
    main()
