"""
Unit tests for TORCS Racing AI components.
"""

import unittest
import torch
import numpy as np
from unittest.mock import Mock, patch

from torcs_ai.ml_models import MLRacingAI, RacingNetwork
from torcs_ai.client import Client, ServerState, DriverAction
from torcs_ai.visualization import RacingVisualizer


class TestMLModels(unittest.TestCase):
    """Test ML model functionality."""

    def setUp(self):
        self.ai = MLRacingAI()

    def test_network_creation(self):
        """Test neural network initialization."""
        self.assertIsInstance(self.ai.steer_net, RacingNetwork)
        self.assertIsInstance(self.ai.accel_net, RacingNetwork)
        self.assertIsInstance(self.ai.brake_net, RacingNetwork)

    def test_predict_action(self):
        """Test action prediction."""
        sensor_data = {
            'speedX': 100.0,
            'angle': 0.1,
            'trackPos': 0.0,
            'track': [0.0] * 19
        }

        actions = self.ai.predict_action(sensor_data)
        self.assertIsInstance(actions, dict)
        self.assertIn('steer', actions)
        self.assertIn('accel', actions)
        self.assertIn('brake', actions)

        # Check value ranges
        self.assertTrue(-1 <= actions['steer'] <= 1)
        self.assertTrue(0 <= actions['accel'] <= 1)
        self.assertTrue(0 <= actions['brake'] <= 1)

    def test_calculate_reward(self):
        """Test reward calculation."""
        sensor_data = {
            'speedX': 150.0,
            'angle': 0.0,
            'trackPos': 0.0,
            'damage': 0
        }
        actions = {'steer': 0.0, 'accel': 0.8, 'brake': 0.0}

        reward = self.ai.calculate_reward(sensor_data, actions)
        self.assertIsInstance(reward, float)
        self.assertGreater(reward, 0)  # Should be positive for good driving


class TestClient(unittest.TestCase):
    """Test TORCS client functionality."""

    def test_server_state_parsing(self):
        """Test server state parsing."""
        ss = ServerState()
        test_string = "(angle 0.1)(speedX 100.5)(trackPos 0.0)"

        ss.parse_server_str(test_string)

        self.assertAlmostEqual(ss.d['angle'], 0.1)
        self.assertAlmostEqual(ss.d['speedX'], 100.5)
        self.assertAlmostEqual(ss.d['trackPos'], 0.0)

    def test_driver_action_repr(self):
        """Test driver action string representation."""
        da = DriverAction()
        da.d['steer'] = 0.5
        da.d['accel'] = 0.8

        action_str = repr(da)
        self.assertIn('(steer', action_str)
        self.assertIn('(accel', action_str)
        self.assertIn('0.500', action_str)
        self.assertIn('0.800', action_str)

    def test_driver_action_clipping(self):
        """Test action value clipping."""
        da = DriverAction()
        da.d['steer'] = 2.0  # Out of range
        da.d['accel'] = -0.5  # Out of range
        da.d['brake'] = 1.5  # Out of range

        da.clip_to_limits()

        self.assertEqual(da.d['steer'], 1.0)
        self.assertEqual(da.d['accel'], 0.0)
        self.assertEqual(da.d['brake'], 1.0)


class TestVisualization(unittest.TestCase):
    """Test visualization functionality."""

    def setUp(self):
        self.viz = RacingVisualizer(max_history=100)

    def test_data_collection(self):
        """Test data collection."""
        sensor_data = {'speedX': 100.0, 'angle': 0.0, 'trackPos': 0.0}
        actions = {'steer': 0.0, 'accel': 0.8, 'brake': 0.0}
        reward = 1.5

        initial_length = len(self.viz.performance_data)
        self.viz.collect_data(sensor_data, actions, reward)

        self.assertEqual(len(self.viz.performance_data), initial_length + 1)

        # Check data integrity
        latest = self.viz.performance_data[-1]
        self.assertEqual(latest['speed'], 100.0)
        self.assertEqual(latest['reward'], 1.5)

    def test_performance_report(self):
        """Test performance report generation."""
        # Add some test data
        for i in range(10):
            sensor_data = {
                'speedX': 100 + i * 10,
                'angle': 0.0,
                'trackPos': 0.0,
                'damage': 0
            }
            actions = {'steer': 0.0, 'accel': 0.8, 'brake': 0.0}
            reward = 1.0 + i * 0.1
            self.viz.collect_data(sensor_data, actions, reward)

        report = self.viz.generate_performance_report()
        self.assertIsInstance(report, str)
        self.assertIn('Performance Report', report)
        self.assertIn('Data Points: 10', report)


class TestUtils(unittest.TestCase):
    """Test utility functions."""

    def test_analyze_track_curvature(self):
        """Test track curvature analysis."""
        from torcs_ai.utils import analyze_track_curvature

        sensor_data = {
            'track': [0.0, 0.1, 0.2, 0.1, 0.0, -0.1, -0.2, -0.1,
                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        }

        curvature, curvatures = analyze_track_curvature(sensor_data)
        self.assertIsInstance(curvature, float)
        self.assertIsInstance(curvatures, list)

    def test_detect_racing_scenarios(self):
        """Test scenario detection."""
        from torcs_ai.utils import detect_racing_scenarios

        # Normal scenario
        sensor_data = {
            'speedX': 150.0,
            'angle': 0.0,
            'trackPos': 0.0,
            'damage': 0,
            'fuel': 50,
            'opponents': [200] * 36  # No nearby opponents
        }

        scenarios = detect_racing_scenarios(sensor_data)
        self.assertIsInstance(scenarios, dict)
        self.assertFalse(scenarios['emergency'])
        self.assertFalse(scenarios['wall_proximity'])

        # Emergency scenario
        sensor_data['trackPos'] = 1.5  # Off track
        sensor_data['damage'] = 1000

        scenarios = detect_racing_scenarios(sensor_data)
        self.assertTrue(scenarios['emergency'])
        self.assertTrue(scenarios['wall_proximity'])


if __name__ == '__main__':
    # Set up test environment
    torch.manual_seed(42)
    np.random.seed(42)

    unittest.main(verbosity=2)