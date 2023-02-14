from pettingzoo.butterfly import cooperative_pong_v5
from supersuit import color_reduction_v0, frame_stack_v1, resize_v1
import argparse
import os
import cv2 as cv
import torch
import numpy as np

# Metadata
stack_size = 4
frame_size = (64, 64)
max_cycles = 125
render_mode = "rgb_array"
parralel = True
color_reduc = True

def coop_pong_env_build(stack_size: int = stack_size, frame_size: tuple = frame_size,
                        max_cycles: int = max_cycles, render_mode: str = render_mode,
                        parralel: bool = parralel, color_reduc: bool = color_reduc):
    """Environment Making

    Args:
        stack_size (int, optional): Number of frames stacked. Defaults to stack_size.
        frame_size (tuple, optional): Frame size. Defaults to frame_size.
        max_cycles (int, optional): after max_cycles steps all agents will return done. Defaults to max_cycles.
        render_mode (str, optional): Type of render. Defaults to render_mode.
        parralel (bool, optional): Let env run on parralel or not. Defaults to parralel.
        color_reduc (bool, optional): Reduce the color channel. Defaults to color_reduc.

    Returns:
        _type_: Environment
    """

    if parralel:
        env = cooperative_pong_v5.parallel_env(render_mode=render_mode, 
                            max_cycles=max_cycles)
    else:
        env = cooperative_pong_v5.env(render_mode=render_mode, 
                            max_cycles=max_cycles)
    
    if color_reduc:
        env = color_reduction_v0(env)
    
    env = resize_v1(env, frame_size[0], frame_size[1])

    if stack_size > 1:
        env = frame_stack_v1(env, stack_size=stack_size)
    env.reset()

    return env

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--stacksize", type=int, default=4,
                        help="")
    parser.add_argument("--framesize", type=tuple, default=(64, 64),
                        help="")
    parser.add_argument("--maxcycles", type=int, default=125,
                        help="")
    parser.add_argument("--rendermode", type=str, default='rgb_array', 
                        choices = ["rgb_array", "human"],
                        help="")
    parser.add_argument("--parralel", type=bool, default=True, 
                        choices = [False, True],
                        help="")
    parser.add_argument("--colorreduc", type=bool, default=True, 
                        choices = [False, True],
                        help="")
    
    args = parser.parse_args()

    env = coop_pong_env_build(stack_size = args.stacksize, frame_size = args.framesize,
                        max_cycles = args.maxcycles, render_mode = args.rendermode,
                        parralel = args.parralel, color_reduc= args.colorreduc)
    
    print("=" * 80)
    print("Summary of warlords env metadata:")
    print(f"Stack size: {args.stacksize}")
    print(f"Frame size: {args.framesize}")
    print(f"Max cycles: {args.maxcycles}")
    print(f"Render mode: {args.rendermode}")
    print(f"Parallel env computing: {args.parralel}")
    print(f"Color reduction: {args.parralel}")
    print("=" * 80)
    print(f"Number of possible agents: {len(env.possible_agents)}")
    print(f"Example of agent: {env.possible_agents[0]}")
    print(f"Number of actions: {env.action_space(env.possible_agents[0]).n}")
    print(f"Action Space: {env.action_space(env.possible_agents[0])}")
    print(f"Observation Size: {env.observation_space(env.possible_agents[0]).shape}")
    
    env.reset()

    # render_array = env.render()
    # cv.imwrite(os.getcwd() + "/envs/pong/render.jpg", render_array)

    # actions = {a : env.action_space(a).sample() for a in env.possible_agents}
    # print("Action: {}".format(actions))

    # agents = env.possible_agents
    # for agent in agents:
    #     for i in range(10):
    #         actions = {a : env.action_space(a).sample() for a in env.possible_agents}
    #         observation, reward, termination, truncation, info = env.step(actions)
    #     obs = observation[agent]
    #     cv.imwrite(os.getcwd() + f"/envs/pong/obs_{agent}.jpg", obs)

    observation = 0
    for i in range(2000):
        render_array = env.render()
        actions = {
            'paddle_0': env.action_space('paddle_0').sample(), 
            'paddle_1': env.action_space('paddle_0').sample()
        }
        observation, reward, termination, truncation, info = env.step(actions)