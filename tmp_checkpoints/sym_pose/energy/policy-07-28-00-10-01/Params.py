# coding: utf8
"""This class stores parameters about the replay"""
from pickle import TRUE
import numpy as np


class Params:
    def __init__(self):
        # Replay parameters
        self.replay_path = "demo_replay.npz"
        self.SIMULATION = False  # Run the replay in simulation if True
        self.LOGGING = True  # Save the logs of the experiments if True
        self.PLOTTING = True  # Plot the logs of the experiments if True

        # Control parameters
        self.dt = 0.001  # Time step of the replay
        self.Kp = np.array([6.0, 6.0, 6.0])  # Proportional gains for the PD+
        self.Kd = np.array([0.3, 0.3, 0.3])  # Derivative gains for the PD+
        self.Kff = 1.0  # Feedforward torques multiplier for the PD+

        # Other parameters
        self.config_file = "config_solo12.yaml"  #  Name of the yaml file containing hardware information


class RLParams:
    def __init__(self):
        # Replay parameters
        self.SIMULATION = True  # Run the replay in simulation if True
        self.LOGGING = True  # Save the logs of the experiments if True
        self.PLOTTING = True  # Save the logs of the experiments if True
        self.max_steps = 60000

        # Control parameters
        self.dt = 0.001
        self.control_dt = 0.01
        self.Kp = np.array([3.0, 3.0, 3.0] * 4)  # Proportional gains for the PD+
        self.Kd = np.array([0.2, 0.2, 0.2] * 4)  # Derivative gains for the PD+
        self.Kff = 0.0  # Feedforward torques multiplier for the PD+

        self.enable_lateral_vel = True
        self.symmetric_pose = True

        self.q_init = np.array(
            [-0.04, 0.7, -1.4, 0.04, 0.7, -1.4, -0.04, 0.7, -1.4, 0.04, 0.7, -1.4]
        )

        if self.symmetric_pose:
            self.q_init = np.array(
                [-0.04, 0.8, -1.6, 0.04, 0.8, -1.6, -0.04, -0.8, 1.6, 0.04, -0.8, 1.6]
            )

        self.alpha = 0.0  # 3569

        # Other parameters
        self.config_file = "config_solo12.yaml"  #  Name of the yaml file containing hardware information
        self.record_video = False
