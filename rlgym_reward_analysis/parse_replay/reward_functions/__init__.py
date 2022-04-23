from . import common_rewards, custom_rewards, extra_rewards

rewards_names_map = {"liu_dist_ball2goal": custom_rewards.liu_dist_ball2goal,
                     "signed_liu_dist_ball2goal": custom_rewards.signed_liu_dist_ball2goal,
                     "velocity_ball2goal": common_rewards.velocity_ball2goal,
                     "ball_y_coord": custom_rewards.ball_y_coord,
                     "velocity": common_rewards.velocity,
                     "save_boost": common_rewards.save_boost,
                     "align_ball": common_rewards.align_ball,
                     "dist_weighted_align_ball": custom_rewards.dist_weighted_align_ball,
                     "offensive_potential": custom_rewards.offensive_potential,
                     "liu_dist_player2ball": custom_rewards.liu_dist_player2ball,
                     "velocity_player2ball": common_rewards.velocity_player2ball,
                     "face_ball": common_rewards.face_ball,
                     "touch_ball": common_rewards.touch_ball,
                     "kickoff": extra_rewards.kickoff,
                     }
