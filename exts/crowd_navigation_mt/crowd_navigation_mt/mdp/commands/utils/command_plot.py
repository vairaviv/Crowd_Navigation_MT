from __future__ import annotations

import matplotlib
# matplotlib.use("Qt5Agg")

import matplotlib.pyplot as plt
import torch

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from crowd_navigation_mt.mdp.commands import RobotGoalCommand


class RobotGoalCommandPlot:

    def __init__(self, xlim=(-10, 10), ylim=(-10, 10)):
        """
        Initialize the plot with limits and labels for robot and goal points.
        
        Args:
            xlim (tuple): Limits for the x-axis.
            ylim (tuple): Limits for the y-axis.
        """
        # Initialize the plot
        self.fig, self.ax = plt.subplots()
        self.robot_point, = self.ax.plot([], [], 'bo', label='Robot')  # Blue dot for robot
        self.goal_point, = self.ax.plot([], [], 'ro', label='Goal')    # Red dot for goal
        self.ax.set_xlim(*xlim)
        self.ax.set_ylim(*ylim)
        self.ax.legend()
        plt.ion()  # Enable interactive mode for real-time updates
        plt.show()

    def _plot(self, command: RobotGoalCommand):
        # Move tensors to the CPU and convert to numpy before plotting, requirements from matplotlib
        robot_state = command.pos_spawn_w.cpu().numpy()  
        goal_state = command.pos_command_w.cpu().numpy()  
        self.robot_point.set_data(robot_state[0], robot_state[1])
        self.goal_point.set_data(goal_state[0], goal_state[1])
        plt.draw()
        plt.pause(0.01)  # Pause to refresh the plot
