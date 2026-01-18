"""
Advanced Racing Visualization Module

Provides comprehensive visualization of racing data, ML model performance,
and real-time telemetry using modern plotting libraries.
"""

import time
import logging
from typing import Dict, List, Any, Optional
from collections import deque
import numpy as np

logger = logging.getLogger(__name__)

try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("matplotlib not available, visualization disabled")

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logger.warning("plotly not available, advanced visualization disabled")


class RacingVisualizer:
    """Advanced visualizer for racing data and ML performance."""

    def __init__(self, max_history: int = 2000):
        self.performance_data: deque = deque(maxlen=max_history)
        self.sensor_history: deque = deque(maxlen=max_history)
        self.realtime_data: deque = deque(maxlen=100)  # For live updates
        self.max_history = max_history

        # Real-time plotting
        self.fig: Optional[Figure] = None
        self.ani: Optional[animation.FuncAnimation] = None

    def collect_data(self, sensors: Dict[str, Any], actions: Dict[str, float], reward: float) -> None:
        """Collect racing data for visualization."""
        data_point = {
            'speed': sensors.get('speedX', 0),
            'angle': sensors.get('angle', 0),
            'track_pos': sensors.get('trackPos', 0),
            'steer': actions.get('steer', 0),
            'accel': actions.get('accel', 0),
            'brake': actions.get('brake', 0),
            'reward': reward,
            'damage': sensors.get('damage', 0),
            'fuel': sensors.get('fuel', 100),
            'rpm': sensors.get('rpm', 0),
            'gear': sensors.get('gear', 0),
            'timestamp': time.time()
        }

        self.performance_data.append(data_point)
        self.sensor_history.append(data_point)
        self.realtime_data.append(data_point)

    def plot_comprehensive_analysis(self, save_path: Optional[str] = None) -> None:
        """Create comprehensive analysis plots."""
        if not MATPLOTLIB_AVAILABLE or len(self.sensor_history) < 100:
            return

        try:
            fig = plt.figure(figsize=(16, 12))
            gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)

            # Extract data
            data = list(self.sensor_history)
            speeds = [d['speed'] for d in data]
            angles = [d['angle'] for d in data]
            track_positions = [d['track_pos'] for d in data]
            steers = [d['steer'] for d in data]
            rewards = [d['reward'] for d in data]
            damages = [d['damage'] for d in data]

            # Speed distribution and time series
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.hist(speeds, bins=30, alpha=0.7, color='blue', density=True)
            ax1.set_title('Speed Distribution')
            ax1.set_xlabel('Speed (km/h)')

            ax2 = fig.add_subplot(gs[0, 1:])
            ax2.plot(speeds, alpha=0.7, color='blue')
            ax2.set_title('Speed Over Time')
            ax2.set_xlabel('Time Steps')
            ax2.set_ylabel('Speed (km/h)')
            ax2.grid(True, alpha=0.3)

            # Steering analysis
            ax3 = fig.add_subplot(gs[1, 0])
            ax3.scatter(track_positions, steers, alpha=0.5, s=10, color='green')
            ax3.set_title('Track Pos vs Steering')
            ax3.set_xlabel('Track Position')
            ax3.set_ylabel('Steering')

            ax4 = fig.add_subplot(gs[1, 1])
            ax4.scatter(angles, steers, alpha=0.5, s=10, color='orange')
            ax4.set_title('Angle vs Steering')
            ax4.set_xlabel('Angle (rad)')
            ax4.set_ylabel('Steering')

            ax5 = fig.add_subplot(gs[1, 2:])
            ax5.plot(steers, alpha=0.7, color='green')
            ax5.set_title('Steering Over Time')
            ax5.set_xlabel('Time Steps')
            ax5.set_ylabel('Steering')
            ax5.grid(True, alpha=0.3)

            # Performance metrics
            ax6 = fig.add_subplot(gs[2, :2])
            ax6.plot(rewards, alpha=0.7, color='purple')
            ax6.set_title('Reward Over Time')
            ax6.set_xlabel('Time Steps')
            ax6.set_ylabel('Reward')
            ax6.grid(True, alpha=0.3)

            ax7 = fig.add_subplot(gs[2, 2:])
            ax7.plot(damages, alpha=0.7, color='red')
            ax7.set_title('Damage Over Time')
            ax7.set_xlabel('Time Steps')
            ax7.set_ylabel('Damage')
            ax7.grid(True, alpha=0.3)

            # Track position stability
            ax8 = fig.add_subplot(gs[3, :])
            abs_track_pos = [abs(tp) for tp in track_positions]
            ax8.plot(abs_track_pos, alpha=0.7, color='red')
            ax8.axhline(y=1, color='r', linestyle='--', alpha=0.7, label='Track Edge')
            ax8.fill_between(range(len(abs_track_pos)), 0, 1, alpha=0.1, color='green', label='Safe Zone')
            ax8.fill_between(range(len(abs_track_pos)), 1, max(abs_track_pos), alpha=0.1, color='red', label='Danger Zone')
            ax8.set_title('Track Position Stability')
            ax8.set_xlabel('Time Steps')
            ax8.set_ylabel('Distance from Center')
            ax8.legend()
            ax8.grid(True, alpha=0.3)

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                logger.info(f"Comprehensive analysis saved to {save_path}")
            else:
                plt.savefig('racing_comprehensive_analysis.png', dpi=150, bbox_inches='tight')
                logger.info("Comprehensive analysis saved as 'racing_comprehensive_analysis.png'")

            plt.close()

        except Exception as e:
            logger.error(f"Error creating comprehensive analysis plot: {e}")

    def create_interactive_dashboard(self, save_path: Optional[str] = None) -> None:
        """Create interactive dashboard using plotly."""
        if not PLOTLY_AVAILABLE or len(self.sensor_history) < 50:
            return

        try:
            data = list(self.sensor_history)

            # Create subplots
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=('Speed & Reward Over Time', 'Steering Analysis',
                              'Track Position Stability', 'Performance Metrics',
                              'Sensor Correlations', 'Damage & Fuel'),
                specs=[[{'secondary_y': True}, {}],
                       [{}, {}],
                       [{'type': 'scatter3d'}, {}]]
            )

            timestamps = [d['timestamp'] for d in data]
            speeds = [d['speed'] for d in data]
            rewards = [d['reward'] for d in data]
            steers = [d['steer'] for d in data]
            track_pos = [d['track_pos'] for d in data]
            angles = [d['angle'] for d in data]
            damages = [d['damage'] for d in data]
            fuels = [d['fuel'] for d in data]

            # Speed and reward
            fig.add_trace(go.Scatter(x=timestamps, y=speeds, name='Speed', line=dict(color='blue')), row=1, col=1)
            fig.add_trace(go.Scatter(x=timestamps, y=rewards, name='Reward', line=dict(color='purple'), yaxis='y2'), row=1, col=1, secondary_y=True)

            # Steering analysis
            fig.add_trace(go.Scatter(x=track_pos, y=steers, mode='markers', name='Track Pos vs Steer',
                                   marker=dict(size=4, opacity=0.6)), row=1, col=2)

            # Track position stability
            fig.add_trace(go.Scatter(x=timestamps, y=[abs(tp) for tp in track_pos], name='Track Position',
                                   fill='tozeroy', line=dict(color='red')), row=2, col=1)
            fig.add_hline(y=1, line_dash='dash', line_color='red', row=2, col=1)

            # Performance metrics
            fig.add_trace(go.Scatter(x=timestamps, y=rewards, name='Reward', line=dict(color='green')), row=2, col=2)
            fig.add_trace(go.Scatter(x=timestamps, y=damages, name='Damage', line=dict(color='orange')), row=2, col=2)

            # 3D correlation plot
            fig.add_trace(go.Scatter3d(x=speeds, y=angles, z=steers, mode='markers',
                                     marker=dict(size=3, color=rewards, colorscale='Viridis',
                                               showscale=True, colorbar=dict(title='Reward')),
                                     name='Speed-Angle-Steer'), row=3, col=1)

            # Damage and fuel
            fig.add_trace(go.Scatter(x=timestamps, y=damages, name='Damage', line=dict(color='red')), row=3, col=2)
            fig.add_trace(go.Scatter(x=timestamps, y=fuels, name='Fuel', line=dict(color='blue')), row=3, col=2)

            fig.update_layout(height=1000, title_text="TORCS Racing AI Dashboard")
            fig.update_xaxes(title_text="Time", row=1, col=1)
            fig.update_xaxes(title_text="Track Position", row=1, col=2)
            fig.update_xaxes(title_text="Time", row=2, col=1)
            fig.update_xaxes(title_text="Time", row=2, col=2)
            fig.update_xaxes(title_text="Time", row=3, col=2)

            if save_path:
                fig.write_html(save_path)
                logger.info(f"Interactive dashboard saved to {save_path}")
            else:
                fig.write_html('racing_dashboard.html')
                logger.info("Interactive dashboard saved as 'racing_dashboard.html'")

        except Exception as e:
            logger.error(f"Error creating interactive dashboard: {e}")

    def plot_realtime_telemetry(self) -> None:
        """Create real-time telemetry visualization."""
        if not MATPLOTLIB_AVAILABLE:
            return

        try:
            if self.fig is None:
                self.fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
                self.fig.suptitle('Real-time TORCS Telemetry')

                # Initialize plots
                self.speed_line, = ax1.plot([], [], 'b-', label='Speed')
                ax1.set_title('Speed')
                ax1.set_xlabel('Time')
                ax1.set_ylabel('km/h')
                ax1.legend()

                self.steer_line, = ax2.plot([], [], 'g-', label='Steering')
                ax2.set_title('Steering')
                ax2.set_xlabel('Time')
                ax2.set_ylabel('Steering Angle')
                ax2.legend()

                self.track_pos_line, = ax3.plot([], [], 'r-', label='Track Position')
                ax3.set_title('Track Position')
                ax3.set_xlabel('Time')
                ax3.set_ylabel('Position')
                ax3.axhline(y=1, color='r', linestyle='--', alpha=0.5)
                ax3.axhline(y=-1, color='r', linestyle='--', alpha=0.5)
                ax3.legend()

                self.reward_line, = ax4.plot([], [], 'purple', label='Reward')
                ax4.set_title('Reward')
                ax4.set_xlabel('Time')
                ax4.set_ylabel('Reward')
                ax4.legend()

            # Update data
            if len(self.realtime_data) > 0:
                data = list(self.realtime_data)
                x_data = range(len(data))

                speeds = [d['speed'] for d in data]
                steers = [d['steer'] for d in data]
                track_pos = [d['track_pos'] for d in data]
                rewards = [d['reward'] for d in data]

                self.speed_line.set_data(x_data, speeds)
                self.steer_line.set_data(x_data, steers)
                self.track_pos_line.set_data(x_data, track_pos)
                self.reward_line.set_data(x_data, rewards)

                # Adjust axes limits
                for ax in [self.fig.axes[0], self.fig.axes[1], self.fig.axes[2], self.fig.axes[3]]:
                    ax.relim()
                    ax.autoscale_view()

            plt.pause(0.01)  # Small pause for real-time effect

        except Exception as e:
            logger.error(f"Error in real-time telemetry: {e}")

    def generate_performance_report(self) -> str:
        """Generate a text-based performance report."""
        if len(self.performance_data) < 10:
            return "Insufficient data for performance report"

        data = list(self.performance_data)
        speeds = [d['speed'] for d in data]
        rewards = [d['reward'] for d in data]
        damages = [d['damage'] for d in data]
        track_positions = [abs(d['track_pos']) for d in data]

        report = f"""
TORCS Racing AI Performance Report
===================================

Data Points: {len(data)}
Time Span: {data[-1]['timestamp'] - data[0]['timestamp']:.1f} seconds

Speed Statistics:
- Average Speed: {np.mean(speeds):.1f} km/h
- Max Speed: {np.max(speeds):.1f} km/h
- Min Speed: {np.min(speeds):.1f} km/h
- Speed Consistency (std): {np.std(speeds):.1f}

Performance Metrics:
- Average Reward: {np.mean(rewards):.3f}
- Total Reward: {np.sum(rewards):.1f}
- Best Reward: {np.max(rewards):.3f}
- Reward Stability (std): {np.std(rewards):.3f}

Safety Metrics:
- Average Damage: {np.mean(damages):.1f}
- Max Damage: {np.max(damages):.1f}
- Off-track Incidents: {sum(1 for tp in track_positions if tp > 1.0)}
- Track Position Stability: {np.mean(track_positions):.3f}

Efficiency:
- Final Fuel Level: {data[-1]['fuel']:.1f}
- Fuel Efficiency: {(data[0]['fuel'] - data[-1]['fuel']) / len(data) * 1000:.3f} fuel/1000 steps
"""

        return report

    def save_data_to_csv(self, filename: str = 'racing_data.csv') -> None:
        """Save collected data to CSV for further analysis."""
        try:
            import csv

            if len(self.performance_data) == 0:
                return

            fieldnames = ['timestamp', 'speed', 'angle', 'track_pos', 'steer', 'accel', 'brake',
                         'reward', 'damage', 'fuel', 'rpm', 'gear']

            with open(filename, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for data_point in self.performance_data:
                    writer.writerow({k: v for k, v in data_point.items() if k in fieldnames})

            logger.info(f"Data saved to {filename}")

        except Exception as e:
            logger.error(f"Error saving data to CSV: {e}")


# Global visualizer instance
visualizer = RacingVisualizer()