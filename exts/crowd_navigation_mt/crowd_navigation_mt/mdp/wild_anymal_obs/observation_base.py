# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
#from abc import classmethod


class ObservationBase:
    def __init__(self, simulation_dt=0.0025, control_dt=0.02, is_training=False):
        self.simulation_dt = simulation_dt
        self.control_dt = control_dt
        self.is_training = is_training

    @classmethod
    def substep_update(self, **kwargs):
        """Update State for each substep"""
        raise NotImplementedError

    @classmethod
    def update(self, **kwargs):
        """Update observation"""
        raise NotImplementedError

    def get_obs(self) -> torch.Tensor:
        return (self.obs - self.mean) / self.std

    @property
    def num_obs(self) -> int:
        return self.obs.shape[1]
