import numpy as np
from rlgym.utils import common_values

from rlgym_reward_analysis.parse_replay import _objectives


def velocity_ball2goal(frames, player_team):
    objective = _objectives[int(player_team[1])]

    pos_diff = objective - frames['ball'][['pos_x', 'pos_y', 'pos_z']]
    norm_pos_diff = pos_diff / (np.linalg.norm(pos_diff, axis=-1)[..., None] + 1e-8)
    ball_lin_velocity = frames['ball'][['vel_x', 'vel_y', 'vel_z']] / (common_values.BALL_MAX_SPEED * 10)
    return (norm_pos_diff.values * ball_lin_velocity.values).sum(1)
