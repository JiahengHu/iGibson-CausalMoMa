from igibson.tasks.multistep_reaching_random_task import MultistepReachingRandomTask
import numpy as np

class FactoredMultistepReachingRandomTask(MultistepReachingRandomTask):
    """
    Reaching Random Task
    The goal is to reach a random goal position with the robot's end effector
    """

    def __init__(self, env):
        super(FactoredMultistepReachingRandomTask, self).__init__(env)

    def get_reward(self, env, collision_links=[], action=None, info={}):
        """
        Key change is that we return factored reward

        :param env: environment instance
        :param collision_links: collision links after executing action
        :param action: the executed action
        :param info: additional info
        :return reward: total reward of the current timestep
        :return info: additional info
        """
        r_list, info = self.get_reward_list(env, collision_links, action, info)
        return np.array(r_list), info