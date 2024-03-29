import numpy as np

from . import common_rewards
from .. import _objectives, _common_values

_goal_depth = _common_values.BACK_NET_Y - _common_values.BACK_WALL_Y + _common_values.BALL_RADIUS


def liu_dist_ball2goal(frames, player_team, dispersion=1., density=1., own_goal=False):
    ball_position = frames['ball'][['pos_x', 'pos_y', 'pos_z']]
    if own_goal:
        objective = _objectives[::-1][int(player_team[1])]
    else:
        objective = _objectives[int(player_team[1])]

    dist = np.linalg.norm(ball_position - objective, axis=-1) - _goal_depth  # adjusted by goal depth radius
    rew = np.exp(-0.5 * dist / (_common_values.BALL_MAX_SPEED * dispersion))  # with dispersion
    rew **= (1 / density)  # with density

    return rew


def signed_liu_dist_ball2goal(frames, player_team, dispersion=1., density=1.):
    ball_position = frames['ball'][['pos_x', 'pos_y', 'pos_z']]
    objective = _objectives[int(player_team[1])]

    dist = np.linalg.norm(ball_position - objective, axis=-1) - _goal_depth  # adjusted by goal depth radius
    # 4570: trigonometry solution - produces an approximate unsigned value of 0.5 at position [4096, 0, 93]
    rew = np.exp(-0.5 * dist / (4570 * dispersion))  # with dispersion
    rew = (rew - 0.5) * 2  # signed
    rew = (np.abs(rew) ** (1 / density)) * np.sign(rew)  # with density

    return rew


def liu_dist_ball2goal_diff(frames,
                            player_team,
                            off_dispersion=0.5,
                            def_dispersion=0.5,
                            def_density=1.,
                            off_density=1.,
                            off_weight=1.,
                            def_weight=1.):
    return (off_weight * liu_dist_ball2goal(frames, player_team, off_dispersion, off_density) -
            def_weight * liu_dist_ball2goal(frames, player_team, def_dispersion, def_density, True))


def ball_y_coord(frames, player_team, exponent=1.):
    ball_y_position = frames['ball']['pos_y'].values
    if int(player_team[1]):
        ball_y_position *= -1
    rew = ball_y_position / (_common_values.BACK_WALL_Y + _common_values.BALL_RADIUS)
    rew = (np.abs(rew) ** exponent) * np.sign(rew)
    return rew


def liu_dist_player2ball(frames, player_team, dispersion=1., density=1.):
    ball_position = frames['ball'][['pos_x', 'pos_y', 'pos_z']]
    player_position = frames[player_team[0]][['pos_x', 'pos_y', 'pos_z']]

    dist = np.linalg.norm(player_position - ball_position, axis=-1) - _common_values.BALL_RADIUS
    return np.exp(-0.5 * dist / (_common_values.CAR_MAX_SPEED * dispersion)) ** (1 / density)


def dist_weighted_align_ball(frames,
                             player_team,
                             defense=0.5,
                             offense=0.5,
                             dispersion=1.,
                             density=1.):
    align_ball_rew = common_rewards.align_ball(frames, player_team, defense, offense)
    liu_dist_player2ball_rew = liu_dist_player2ball(frames, player_team, dispersion, density)

    rew = align_ball_rew * liu_dist_player2ball_rew
    # "weighted" product (n_1 * n_2 * ... * n_N) ^ (1 / N)
    return np.sqrt(np.abs(rew)) * np.sign(rew)


def offensive_potential(frames,
                        player_team,
                        defense=0.5,
                        offense=0.5,
                        dispersion=1.,
                        density=1.):
    velocity_player2ball_rew = common_rewards.velocity_player2ball(frames, player_team)
    align_ball_rew = common_rewards.align_ball(frames, player_team, defense, offense)
    liu_dist_player2ball_rew = liu_dist_player2ball(frames, player_team, dispersion, density)

    # logical AND
    # when both alignment and player to ball velocity are negative we must get a negative output
    # no need to compute for liu_dist_player2ball_rew, positive only
    sign = (((velocity_player2ball_rew >= 0) & (align_ball_rew >= 0)) - 0.5) * 2
    rew = align_ball_rew * velocity_player2ball_rew * liu_dist_player2ball_rew
    # "weighted" product (n_1 * n_2 * ... * n_N) ^ (1 / N)
    return (np.abs(rew) ** (1 / 3)) * sign
