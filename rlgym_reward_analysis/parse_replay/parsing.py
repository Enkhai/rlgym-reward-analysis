from typing import Sequence, Union, Dict, Callable, Tuple

import numpy as np
import pandas as pd

from .reward_functions import rewards_names_map


def parse_replay(df: pd.DataFrame,
                 reward_names_args: Union[None, Sequence[Union[str, Tuple[str, dict]]]] = None,
                 reward_names_fns: Union[None, Dict[str, Callable[[pd.DataFrame, np.ndarray], np.ndarray]]] = None):
    assert reward_names_args or reward_names_fns, "Either `reward_names_args` or `reward_names_fns` must be provided"

    non_players = ['ball', 'game']
    player_names = [c for c in df.columns.levels[0] if c not in non_players]
    # Frame 1: negative coordinate blue (0), positive coordinate orange (1)
    team_idcs = (df[player_names].xs('pos_y', level=1, axis=1).iloc[0] > 0).values.astype(int)
    players_teams = np.stack((player_names, team_idcs), axis=-1)

    if reward_names_fns is None:
        reward_names_fns = {}
        for r in reward_names_args:
            if type(r) is str:
                r_name, r_args = r, {}
            else:
                r_name, r_args = r
            reward_names_fns[r_name] = (lambda frames, player_team:
                                        rewards_names_map[r_name](frames,
                                                                  player_team,
                                                                  **r_args))

    player_reward_values = {(p_t[0], r_n): reward_names_fns[r_n](df, p_t)
                            for p_t in players_teams
                            for r_n in reward_names_fns}

    reward_values_df = pd.DataFrame(player_reward_values)
    assert reward_values_df.shape[0] == df.shape[0]  # assert for same number of rows

    return reward_values_df


def parse_replays(paths,
                  reward_names_args: Union[None, Sequence[Union[str, Tuple[str, dict]]]] = None):
    pass
