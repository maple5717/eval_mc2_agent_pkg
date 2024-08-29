#! /usr/bin/env python

from utils.config_utils import (
    create_agent_config,
    create_env_config,
    get_habitat_config,
    get_omega_config,
)
from mc2.agent.mc2_agent.mc2_agent import MC2Agent
from home_robot.core.interfaces import DiscreteNavigationAction, Observations 

import argparse
import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import rospy
from sensor_msgs.msg import Image
from message_filters import ApproximateTimeSynchronizer, Subscriber
from evaluators import Evaluator

if __name__ == "__main__":
    agent = 0
    if True:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--evaluation_type",
            type=str,
            choices=["local", "local_vectorized", "remote"],
            default="local",
        )
        parser.add_argument("--num_episodes", type=int, default=None)
        parser.add_argument(
            "--habitat_config_path",
            type=str,
            default="ovmm/ovmm_eval.yaml",
            help="Path to config yaml",
        )
        parser.add_argument(
            "--baseline_config_path",
            type=str,
            default="projects/habitat_ovmm/configs/agent/heuristic_agent.yaml",
            help="Path to config yaml",
        )
        parser.add_argument(
            "--env_config_path",
            type=str,
            default="projects/habitat_ovmm/configs/env/hssd_eval.yaml",
            help="Path to config yaml",
        )
        parser.add_argument(
            "--agent_type",
            type=str,
            default="baseline",
            choices=["baseline", "random", "explore"],
            help="Agent to evaluate",
        )
        parser.add_argument(
            "--force_step",
            type=int,
            default=20,
            help="force to switch to new episode after a number of steps",
        )
        parser.add_argument(
            "--data_dir",
            type=str,
            default=None,
            help="whether to save obseration history for data collection",
        )
        parser.add_argument(
            "overrides",
            default=None,
            nargs=argparse.REMAINDER,
            help="Modify config options from command line",
        )
        args = parser.parse_args()

        # get habitat config
        habitat_config, _ = get_habitat_config(
            args.habitat_config_path, overrides=args.overrides
        )

        # get baseline config
        baseline_config = get_omega_config(args.baseline_config_path)

        # get env config
        env_config = get_omega_config(args.env_config_path)

        # merge habitat and env config to create env config
        env_config = create_env_config(
            habitat_config, env_config, evaluation_type=args.evaluation_type
        )

        # merge env config and baseline config to create agent config
        agent_config = create_agent_config(env_config, baseline_config)

        device_id = env_config.habitat.simulator.habitat_sim_v0.gpu_device_id

        # create agent
        agent = MC2Agent(agent_config, device_id=device_id)

    evaluator = Evaluator(agent)
    evaluator.run()

    