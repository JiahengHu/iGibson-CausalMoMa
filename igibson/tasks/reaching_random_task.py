import numpy as np

import logging

from igibson.reward_functions.point_goal_reward import PointGoalReward
from igibson.reward_functions.reaching_goal_reward import ReachingGoalReward
from igibson.reward_functions.collision_reward import CollisionReward
from igibson.tasks.point_nav_random_task import PointNavRandomTask, MAX_TRIALS
from igibson.termination_conditions.point_goal import PointGoal
from igibson.termination_conditions.reaching_goal import ReachingGoal
from igibson.objects.visual_marker import VisualMarker
from igibson.utils.utils import cartesian_to_polar, l2_distance, rotate_vector_3d, restoreState

import pybullet as p
from pyquaternion import Quaternion

log = logging.getLogger(__name__)


class ReachingRandomTask(PointNavRandomTask):
    """
    Reaching Random Task
    The goal is to reach a random goal position with the robot's end effector
    """

    def __init__(self, env):
        super(ReachingRandomTask, self).__init__(env)
        self.target_height_range = self.config.get("target_height_range", [0.0, 1.0])
        assert isinstance(self.termination_conditions[-1], PointGoal)
        self.termination_conditions[-1] = ReachingGoal(self.config)
        assert isinstance(self.reward_functions[-1], PointGoalReward)
        self.reward_functions = [self.reward_functions[0], ReachingGoalReward(self.config)]

        # We require the presence of these added values, so no default provided
        self.rd_target = self.config["rd_target"]
        self.simple_ori = self.config["simple_orientation"]
        self.enum_ori = self.config["enum_orientation"]
        self.position_reward = self.config["position_reward"]
        self.proportional_local_reward = self.config["proportional_local_reward"]
        self.dist_tol = self.config["dist_tol"]
        self.robot_name = self.config["robot_name"]
        self.noisy_goal = self.config.get("noisy_goal", 0.0)
        self.reaching_reward_scale = self.config["reaching_reward_scale"]

        self.vis_ee_target = self.config.get("vis_ee_target", False)
        self.ee_scale = -0.5

        if self.vis_ee_target:
            # Set a visual marker
            cyl_length = 0.3
            self.target_ee_pos_vis_obj = VisualMarker(
                visual_shape=p.GEOM_CYLINDER,
                rgba_color=[1, 0, 0, 0.3],
                radius=0.05,
                length=cyl_length,
            )
            env.simulator.import_object(self.target_ee_pos_vis_obj)
            # The visual object indicating the target location may be visible
            for instance in self.target_pos_vis_obj.renderer_instances:
                instance.hidden = False

    def get_l2_potential(self, env):
        """
        L2 distance to the goal

        :param env: environment instance
        :return: potential based on L2 distance to goal
        """
        return l2_distance(env.robots[0].get_eef_position(), self.target_pos)

    def get_potential(self, env):
        """
        Compute task-specific potential: distance to the goal

        :param env: environment instance
        :return: task potential
        """
        return self.get_l2_potential(env)

    def sample_target(self, env):
        cur_robot_pos = env.robots[0].get_position()
        for _ in range(MAX_TRIALS):
            _, target_pos = env.scene.get_random_point_by_room_type('living_room')
            if env.scene.build_graph:
                _, dist = env.scene.get_shortest_path(
                    self.floor_num, cur_robot_pos[:2], target_pos[:2], entire_path=False
                )
            else:
                dist = l2_distance(cur_robot_pos, target_pos)
            if self.target_dist_min < dist < self.target_dist_max:
                break
        target_pos[2] += np.random.uniform(self.target_height_range[0], self.target_height_range[1])
        self.target_pos = target_pos
        self.target_ee_pos[-1] = self.target_pos[-1]

    def reset_target(self, env):
        """
        Reset the target position without changing the robot state
        This is only used for multistep reaching
        """

        state_id = p.saveState()
        for i in range(MAX_TRIALS):
            self.sample_target(env)
            reset_success = env.test_valid_position(env.robots[0], self.target_pos,
                                                    ignore_self_collision=True, offset=self.target_offset)
            restoreState(state_id)
            if reset_success:
                break

    def sample_initial_pose_and_target_pos(self, env):
        """
        Sample robot initial pose and target position

        :param env: environment instance
        :return: initial pose and target position
        """

        _, initial_pos = env.scene.get_random_point_by_room_type('living_room')
        max_trials = MAX_TRIALS
        dist = 0.0
        for _ in range(max_trials):
            _, target_pos = env.scene.get_random_point_by_room_type('living_room')
            if env.scene.build_graph:
                _, dist = env.scene.get_shortest_path(
                    self.floor_num, initial_pos[:2], target_pos[:2], entire_path=False
                )
            else:
                dist = l2_distance(initial_pos, target_pos)
            if self.target_dist_min < dist < self.target_dist_max:
                break
        if not (self.target_dist_min < dist < self.target_dist_max):
            log.warning("Failed to sample initial and target positions")
        initial_orn = np.array([0, 0, np.random.uniform(0, np.pi * 2)])
        delta_x, delta_y = target_pos[:2] - initial_pos[:2]
        robot_2_target = np.arctan2(delta_y, delta_x)
        initial_orn = np.array([0, 0, robot_2_target])
        log.debug("Sampled initial pose: {}, {}".format(initial_pos, initial_orn))
        log.debug("Sampled target position: {}".format(target_pos))

        target_pos[2] += np.random.uniform(self.target_height_range[0], self.target_height_range[1])
        return initial_pos, initial_orn, target_pos

    def get_task_obs(self, env):
        """
        Get task-specific observation, including goal position, end effector position, etc.

        :param env: environment instance
        :return: task-specific observation
        """

        task_obs = self.global_to_local(env, self.target_pos)  # x, y, z

        # Add noise to the goal location
        if self.noisy_goal > 0:
            # Generate 3d noise
            goal_noise = np.random.normal(loc=0.0, scale=self.noisy_goal, size=3)
            task_obs += goal_noise

        if self.goal_format == "polar":
            task_obs[:2] = np.array(cartesian_to_polar(task_obs[0], task_obs[1]))

        # linear velocity along the x, y axis
        linear_velocity = rotate_vector_3d(env.robots[0].get_linear_velocity(), *env.robots[0].get_rpy())[:2]
        # angular velocity along the z-axis
        angular_velocity = rotate_vector_3d(env.robots[0].get_angular_velocity(), *env.robots[0].get_rpy())[2]
        task_obs = np.append(task_obs, [*linear_velocity, angular_velocity])

        end_effector_pos_local = self.global_to_local(env, env.robots[0].get_eef_position())
        # ee_current_local_pos & ee_target_local_pos
        task_obs = np.append(task_obs, end_effector_pos_local)

        # ee_current orientation & ee_target_orientation
        eef_local_ori = env.robots[0].get_relative_eef_orientation()
        task_obs = np.append(task_obs, eef_local_ori)
        task_obs = np.append(task_obs, self.pb_target_ee_ori)

        # Lastly, append head position
        head_qpos = env.robots[0].joint_positions[env.robots[0].camera_control_idx]
        task_obs = np.append(task_obs, head_qpos)

        if self.robot_name == "HSR":
            # Add joint observations
            arm_qpos = env.robots[0].joint_positions[env.robots[0].arm_control_idx['0']]
            task_obs = np.append(task_obs, arm_qpos)

        # Add current timestep
        task_obs = np.append(task_obs, env.current_step / self.config["max_step"])

        if env.print_reward:
            print(f"current step: {env.current_step}")
        return task_obs

    def get_info(self, env, collision_links=[], action=None, info={}):
        """
        For this task, the info is to obtain collision information
        """

        # Obtain collision information
        info["base_collision"] = False
        info["arm_collision"] = False
        info["self_collision"] = False
        info["gripper_closed"] = False

        for collision_link in collision_links:
            bodyA, bodyB, linkA, linkB = collision_link[1:5]
            linkA_name = env.robots[0].link_idx_to_name[linkA]
            linkA_group = env.robots[0].get_link_group(linkA_name)

            if bodyB == bodyA:
                linkB_name = env.robots[0].link_idx_to_name[linkB]
                linkB_group = env.robots[0].get_link_group(linkB_name)
                if linkB_group == "gripper" and linkA_group == "gripper":
                    info["gripper_closed"] = True
                else:
                    info["self_collision"] = True
            else:
                # Collision with wall / objects
                if linkA_group in {"base", "head"}:
                    info["base_collision"] = True
                elif linkA_group in {"gripper", "arm"}:
                    info["arm_collision"] = True
                else:
                    raise NotImplementedError

        info["collision_occur"] = info["base_collision"] or info["arm_collision"] or info["self_collision"]
        return info

    def get_ee_ori_reward(self, env):
        # Additional reward 1: keep the hand orientation gets a special reward
        local_ori = self.quat_pb_2_np(env.robots[0].get_relative_eef_orientation())

        if self.simple_ori:
            # This would be a vector distance
            local_ori_x = local_ori.rotate([1, 0, 0])
            orientation_dist = np.linalg.norm(np.array(local_ori_x) - self.target_ori_x)
            ori_threshold = 0.25
        else:
            orientation_dist = Quaternion.absolute_distance(self.target_ee_ori, local_ori)
            ori_threshold = 0.3

        if self.proportional_local_reward:
            ori_r = orientation_dist / 2 * self.ee_scale
        else:
            if orientation_dist < ori_threshold:
                ori_r = 0
            else:
                ori_r = self.ee_scale
        return ori_r

    def get_ee_pos_reward(self, env):
        # Additional reward 2: keeping the arm within a certain range of the body
        local_ee_pos = env.robots[0].get_relative_eef_position()
        # ee_pos_dist = np.linalg.norm(local_ee_pos - self.target_ee_pos)
        ee_pos_dist = np.abs(local_ee_pos[-1] - self.target_ee_pos[-1])

        self.ee_pos_scale = -0.5
        # this is to make the training a bit more stable
        self.ee_pos_reward_offset = 0.2

        if self.proportional_local_reward:
            ee_pos_r = ee_pos_dist * self.ee_pos_scale
            ee_pos_r += self.ee_pos_reward_offset
        else:
            ee_pos_threshold = 0.15
            if ee_pos_dist < ee_pos_threshold:
                ee_pos_r = 0
            else:
                ee_pos_r = self.ee_scale
        # if env.print_reward:
        #     print(ee_pos_dist)
        return ee_pos_r

    def get_reaching_reward(self, env, action):
        reaching_r = 0.0
        for reward_function in self.reward_functions:
            # remove collision reward
            if isinstance(reward_function, CollisionReward):
                raise Exception("Reward shouldn't contain collision")
            reward_channel = reward_function.get_reward(self, env)
            reaching_r += reward_channel
        return reaching_r

    def get_obj_tracking_reward(self, env):
        # A reward that encourages the robot to track the object location
        q_pos = env.robots[0].joint_positions[env.robots[0] .camera_control_idx]

        target_pos = self.global_to_local(env, self.target_pos)
        r, h_rot = cartesian_to_polar(target_pos[0], target_pos[1])

        z_eye = env.robots[0]._links['eyes'].get_position()[-1]
        z_relative = self.target_pos[-1] - z_eye
        v_rot = np.arctan2(z_relative, r)

        # vertical: both angles are within +- pi/2, so an easy subtraction would work
        v_rot_diff = np.abs(q_pos[1] - v_rot)

        # horizontal
        h_rot_diff = (q_pos[0] - h_rot) % (2 * np.pi)
        if h_rot_diff > np.pi:
            h_rot_diff -= 2 * np.pi
            h_rot_diff *= -1
        # This will make the fov look more like a circle
        final_dist = np.sqrt(h_rot_diff**2 + v_rot_diff**2)

        threshold = 0.7
        if final_dist > threshold:
            return 0
        else:
            return 0.2


    def get_grasp_reward(self, env, info, action):
        """
        This reward is designed for the multistep env: robot should grasp when inside the target region
        """
        local_z_pos = env.robots[0].get_relative_eef_position()[-1]
        threshold = np.mean(self.config["target_height_range"])  # 0.7
        grasp_action = action[-1]  # last dimension is action
        if env.print_reward:
            print(f"debug grasp reward: z_pos: {local_z_pos}, grasp_action: {grasp_action}, "
                  f"target z: {self.target_pos[2]}")

        return_r = np.abs(grasp_action) * -0.2
        # Range is -0.2 to 0.2
        if local_z_pos >= threshold and grasp_action > 0:
            return_r *= -1
        if local_z_pos < threshold and grasp_action < 0:
            return_r *= -1

        return return_r

    def get_reward_list(self, env, collision_links=[], action=None, info={}):
        """
        Return rewards as a list of reward channels

        :param env: environment instance
        :param collision_links: collision links after executing action
        :param action: the executed action
        :param info: additional info
        :return reward: total reward of the current timestep
        :return info: additional info
        """
        r_list = []

        reaching_r = self.get_reaching_reward(env, action)
        reaching_r *= self.reaching_reward_scale
        r_list.append(reaching_r)

        ori_r = self.get_ee_ori_reward(env)
        r_list.append(ori_r)

        ee_pos_r = 0.0
        if self.position_reward:
            ee_pos_r = self.get_ee_pos_reward(env)
        r_list.append(ee_pos_r)

        # Collision rewards: base, arm, self, gripper
        c_type_list = ["base_collision", "arm_collision", "self_collision", "gripper_closed"]
        collision_weight = -1
        for c_type in c_type_list:
            if info[c_type]:
                c_r = collision_weight
            else:
                c_r = 0
            r_list.append(c_r)

        # A reward for tracking the target, gripper reward will be replaced
        r_list[-1] = self.get_obj_tracking_reward(env)

        grasp_r = self.get_grasp_reward(env, info, action)
        r_list.append(grasp_r)

        if env.print_reward:
            print(f"reward: {r_list}")

        return r_list, info

    def step(self, env):
        super().step(env)
        if self.vis_ee_target:
            self.calc_global_ee_target(env)
            self.target_ee_pos_vis_obj.set_position_orientation(self.global_ee_pos, self.global_ee_viz_marker_ori)

    def get_reward(self, env, collision_links=[], action=None, info={}):
        """
        Aggreate reward functions

        :param env: environment instance
        :param collision_links: collision links after executing action
        :param action: the executed action
        :param info: additional info
        :return reward: total reward of the current timestep
        :return info: additional info
        """

        r_list, info = self.get_reward_list(env, collision_links, action, info)
        reward = 0.0
        for r in r_list:
            reward += r
        return reward, info

    def local_to_global(self, env, pos):
        """
        Convert a 3D point in global frame to agent's local frame

        :param env: environment instance
        :param pos: a 3D point in global frame
        :return: the same 3D point in agent's local frame
        """
        return rotate_vector_3d(np.array(pos), *env.robots[0].get_rpy(), cck=False) + np.array(env.robots[0].get_position())

    def calc_global_ee_target(self, env):
        # # Only if we want the position to match exactly
        # target_ee_pos = np.array([0.55, 0.55, 1.5])
        # self.global_ee_pos = self.local_to_global(env, target_ee_pos)
        self.global_ee_pos = self.local_to_global(env, self.target_ee_pos)
        robot_base_orn = self.quat_pb_2_np(env.robots[0].get_orientation())
        self.global_ee_viz_marker_ori = self.quat_np_2_pb(
            robot_base_orn * self.target_ee_ori * Quaternion([0, 0.707, 0, 0.707]))

    def reset_ee_target(self, env):
        if self.rd_target:
            fully_rand = not self.enum_ori
            if fully_rand:
                self.target_ee_ori = Quaternion.random()
            else:
                rand_quat_list = [Quaternion([0.707, 0, 0, -0.707]),
                                  Quaternion([1, 0, 0, 0]),
                                  Quaternion([0.707, 0, 0, 0.707]),
                                  Quaternion([0.707, 0, -0.707, 0]),
                                  Quaternion([0.707, 0, 0.707, 0]),]
                target_ee_ori_idx = np.random.randint(len(rand_quat_list))
                self.target_ee_ori = rand_quat_list[target_ee_ori_idx]
            uni = np.random.uniform
            self.target_ee_pos = np.array([uni(0.55, 0.9), uni(-0.4, 0.4), uni(0.6, 1.4)])
        else:
            self.target_ee_ori = Quaternion([0.52065462, 0.45743674,  0.53485316, -0.48332447])
            self.target_ee_pos = np.array([0.55, 0, 1])
        self.pb_target_ee_ori = self.quat_np_2_pb(self.target_ee_ori)

        # We make sure the z value of target pos is the same as the z value of the goal
        self.target_ee_pos[-1] = self.global_to_local(env, self.target_pos)[2]

        if self.simple_ori:
            self.target_ori_x = np.array(self.target_ee_ori.rotate([1, 0, 0]))

    def reset_agent(self, env):
        super(ReachingRandomTask, self).reset_agent(env)
        self.reset_ee_target(env)

    def reset_scene(self, env):
        """
        Task-specific scene reset: get a random floor number first

        :param env: environment instance
        """
        super().reset_scene(env)
        # env.scene.randomize_obj_position()

        if "living_room" in self.config["load_room_types"] and self.robot_name == "Fetch":
            env.scene.random_obj_removal()

    def quat_pb_2_np(self, quat):
        """
        Convert a pybullet quaternion to a Quaternion object
        Notice that pybullet quaternion is of [x, y, z, w] while the np version is [w, x, y, z]
        """
        return Quaternion(quat[3], *quat[:3])

    def quat_np_2_pb(self, quat):
        return np.array([quat[1], quat[2], quat[3], quat[0]])


