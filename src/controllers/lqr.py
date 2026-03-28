"""
LQR Controller (single-phase)

Paper 5.1 - Classical Baseline Controller

Notes
-----
Body-frame convention (q=[0, 1,0,0] = nozzle-down base):
  wy=+1 -> nozzle tips East  -> euler_pitch = -R[0,2]  -> use euler[1] = -R[0,2]
  wx=+1 -> nozzle tips North -> euler_roll  = -R[1,2]  -> use euler[0] = -R[1,2]
  wz=+1 -> spin about nozzle axis (yaw)

LQR reduced state (10-dim, DD-008):
 [x, y, z, vx, vy, vz, tilt_EW, tilt_NS, wy, wx]
 tilt_EW = R[0, 2] (East component of nozzle in inertial; +ve = nozzle East)
 tilt_NS = R[1, 2] (North component of nozzle in inertial; +ve = nozzle North)
 wy = omega[1] drives tilt_EW
 wx = omega[0] drives tilt_NS 

Control: u = [throttle, delta_pitch, delta_yaw]
 delta_pitch -> TVC East-West gimbal (M about body-y -> corrects tilt_EW)
 delta_yaw   -> TVC North-South gimbal (M about body-x -> corrects tilt_NS)
"""

import numpy as np 
from scipy.linalg import solve_continuous_are 
from typing import List 
from dataclasses import dataclass 


from dynamics.dynamics import (
    VehicleParams, quat_normalize, quat_to_rotmat,
    equations_of_motion, thrust_force_moment
)


G0 = 9.80665 

# Frame-correct tilt extraction

def get_tilt(q: np.ndarray):
    """
    returns [tilt_EW, tilt_NS] = [R[0,2], R[1, 2]].
    these are the east and north component of the nozzle
    (+ve z body) direction in the inertial frame.

    """
    R = quat_to_rotmat(quat_normalize(q))
    return np.array(R[0,2], R[1, 2])

