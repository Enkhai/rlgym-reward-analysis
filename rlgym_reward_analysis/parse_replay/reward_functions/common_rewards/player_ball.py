import numpy as np
from scipy.spatial.transform import Rotation

from rlgym_reward_analysis import _common_values


def velocity_player2ball(frames, player_team):
    pos_diff = frames['ball'][['pos_x', 'pos_y', 'pos_z']] - frames[player_team[0]][['pos_x', 'pos_y', 'pos_z']]
    norm_pos_diff = pos_diff / (np.linalg.norm(pos_diff, axis=-1)[..., None] + 1e-8)
    player_lin_velocity = frames[player_team[0]][['vel_x', 'vel_y', 'vel_z']] / (_common_values.CAR_MAX_SPEED * 10)
    return (norm_pos_diff.values * player_lin_velocity.values).sum(1)


def face_ball(frames, player_team):
    # euler angles are (pitch, yaw, roll) - should be (roll, pitch, yaw)
    euler_angles = frames[player_team[0]][['rot_z', 'rot_x', 'rot_y']]
    player_forward_vec = Rotation.from_euler('xyz', euler_angles).as_matrix()[:, :, 0]

    pos_diff = frames['ball'][['pos_x', 'pos_y', 'pos_z']] - frames[player_team[0]][['pos_x', 'pos_y', 'pos_z']]
    norm_pos_diff = pos_diff / (np.linalg.norm(pos_diff, axis=-1)[..., None] + 1e-8)
    return (norm_pos_diff.values * player_forward_vec).sum(1)


def touch_ball(frames, player_team):
    # TODO: need to figure out how to compute this
    raise NotImplementedError
