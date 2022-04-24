import numpy as np
from rlgym.utils import common_values

from rlgym_reward_analysis.parse_replay import _objectives
from rlgym_reward_analysis.utils.math import cosine_similarity


def velocity(frames, player_team, negative=False):
    return (np.linalg.norm(frames[player_team[0]][['vel_x', 'vel_y', 'vel_z']], axis=-1) /
            (common_values.CAR_MAX_SPEED * 10) * (1 - 2 * negative))


def save_boost(frames, player_team):
    return np.sqrt(frames[int(player_team[0])]['boost'] / 255)


def align_ball(frames, player_team, defense=1, offense=1):
    if int(player_team[1]):
        blue_goal, orange_goal = _objectives
    else:
        blue_goal, orange_goal = _objectives[::-1]
    blue_goal, orange_goal = _objectives[::int((int(player_team[1]) - 0.5) * 2)]
    ball_position = frames['ball'][['pos_x', 'pos_y', 'pos_z']].values
    player_position = frames[player_team[0]][['pos_x', 'pos_y', 'pos_z']].values

    defensive = defense * cosine_similarity(ball_position - player_position, player_position - blue_goal)
    offensive = offense * cosine_similarity(ball_position - player_position, orange_goal - player_position)

    return defensive + offensive
