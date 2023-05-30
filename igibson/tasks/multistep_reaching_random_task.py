from igibson.tasks.reaching_random_task import ReachingRandomTask
import numpy as np

class MultistepReachingRandomTask(ReachingRandomTask):
    """
    A more advanced environment, the robot is required to reach multiple goal positions
    """

    def __init__(self, env):
        super(MultistepReachingRandomTask, self).__init__(env)
        self.termination_conditions = [self.termination_conditions[index] for index in [1, 2]]
        self.total_holding_reward = 0
        self.total_holding_reward_threshold = 20

    # Reaching reward is changed to holding reward
    def get_reaching_reward(self, env, action):
        assert(len(self.reward_functions) == 2)
        requires_stationary = False
        potential_r = self.reward_functions[0].get_reward(self, env)
        reaching_r = self.reward_functions[1].get_reward(self, env)
        if requires_stationary:
            if np.abs(action[0]) + np.abs(action[1]) > 0.5:
                # Disable reward if robot is still moving
                reaching_r = 0
        self.total_holding_reward += (reaching_r != 0)
        # This value is a bit random
        if self.total_holding_reward_threshold > 0:
            final_reward = potential_r + reaching_r / self.total_holding_reward_threshold * 2
        else:
            final_reward = 0
        return final_reward

    def get_reward_list(self, env, collision_links=[], action=None, info={}):
        reward, info = super().get_reward_list(env, collision_links, action, info)

        # reset goal to a new location
        if self.total_holding_reward >= self.total_holding_reward_threshold:
            self.reset_target(env)
            self.total_holding_reward = 0

        return reward, info