from ..common_rewards import velocity_player2ball


def kickoff(frames, player_team):
    is_kickoff = (frames['ball']['pos_x'] == 0) & (frames['ball']['pos_y'] == 0)
    return is_kickoff.values * velocity_player2ball(frames, player_team)
