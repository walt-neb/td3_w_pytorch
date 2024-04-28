#filename: test.py
import time
import os
import sys
import numpy as np
import robosuite as suite
from robosuite.wrappers import GymWrapper
from td3_torch import Agent_TD3

import cv2

def create_video(frames, filename, fps=30):
    height, width, _ = frames[0].shape
    size = (width, height)
    out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)  # Using mp4v codec
    for frame in frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))  # Write out frame to video
    out.release()
    print('Video saved to {}'.format(filename))


def load_hyperparams(hyp_file):
    params = {}
    with open(hyp_file, 'r') as f:
        for line in f:
            line = line.strip()
            if "=" in line:
                # Parse line
                var_name, var_value = line.split("=")
                var_name = var_name.strip()  # remove leading/trailing white spaces
                var_value = var_value.strip()

                # Attempt to convert variable value to int, float, or leave as string
                try:
                    var_value = int(var_value)
                except ValueError:
                    try:
                        var_value = float(var_value)
                    except ValueError:
                        # If it's neither int nor float, keep as string
                        pass

                # Add the variable to the params dictionary
                params[var_name] = var_value
    return params

def print_parameters(params):
    if not params:
        print("The parameters dictionary is empty.")
        return

    print("*** Training Parameters: ")
    for key, value in params.items():
        print(f"\t\t{key} = {value}")


if __name__ == "__main__":

    # Obtain the hyperparameters from the command line arguments
    if len(sys.argv) != 3:
        print(f'Usage: python train.py [batch_num] [hyp_file]')
        sys.exit(0)
    else:
        batch_num  = sys.argv[1]
        hyp_file = sys.argv[2]
        hyp_file_root = hyp_file.rstrip('.txt')

    # ----------model hyperparameters ----------------------------------------------------
    params = load_hyperparams(hyp_file)
    print_parameters(params)
    env_name = params["env_name"]
    robots=params['robots']
    print(f'robots: {robots}')
    print(f'env_name: {env_name}')
    score_improvement_delta = params["score_improvement_delta"]
    training_patience_threshold = params["training_patience_threshold"]
    early_stop = False


    env = suite.make(
        env_name=params["env_name"],
        robots=robots,
        controller_configs=suite.load_controller_config(default_controller="JOINT_VELOCITY"),
        has_renderer=True,
        use_camera_obs=False,
        horizon=params["horizon"],
        reward_shaping=params["reward_shaping"],
        control_freq=params["control_freq"],
        batch_num=batch_num,
        hyp_file_root=hyp_file_root,
    )

    env = GymWrapper(env)

    input_dims = tuple(env.observation_space.shape)
    n_actions = env.action_space.shape[0]
    print(f'input_dims: {input_dims}, n_actions: {n_actions}')

    agent = Agent_TD3(actor_learning_rate=params["actor_learning_rate"],
                        critic_learning_rate=params["critic_learning_rate"],
                        tau=params["tau"],
                        input_dims=tuple(env.observation_space.shape),
                        env=env,
                        n_actions=env.action_space.shape[0],
                        layer1_size=params["layer1_size"],
                        layer2_size=params["layer2_size"],
                        layer3_size=params["layer3_size"],
                        batch_size=params["batch_size"],
                        gamma=params["gamma"],
                        update_actor_interval = params["update_actor_interval"],
                        warmup   = params["warmup"],
                        max_replay_buffer_size = params["max_replay_buffer_size"],
                        noise = params["noise"]
                        )

    training_identifier = (f'{batch_num}_'
                           f'{env_name}_'
                           f'{hyp_file_root}_'
                          f'cr_{params["actor_learning_rate"]:.2e}_'
                          f'ar_{params["actor_learning_rate"]:.2e}_'
                          f'l1_{params["layer1_size"]}_'
                          f'l2_{params["layer2_size"]}_'
                          f'l3_{params["layer3_size"]}_'
                          f'b_{params["batch_size"]}')
    ckpt_identifier = (f'{batch_num}_'
                       f'{hyp_file_root}')
    log_path = './logs'+f'/{batch_num}_{hyp_file_root}'

    ckpt_file_name = os.path.join('./checkpoints', ckpt_identifier + '.ckpt')
    os.makedirs('./vids', exist_ok=True)

    #checkpoint_file = os.path.join('./checkpoints', training_identifier+ '.ckpt')
    if os.path.isfile(ckpt_file_name) and os.access(ckpt_file_name, os.R_OK):
        print(f'File {ckpt_file_name} exists and is readable')
        try:
            episode_num = agent.load_checkpoint(ckpt_file_name)
        except Exception as e:
            print("Error loading checkpoint:", e)
            print(f'File {ckpt_file_name} either does not exist or is not readable')
            sys.exit(0)
    else:
        print(f'File {ckpt_file_name} either does not exist or is not readable')
        sys.exit(0)

    end_episode = 3
    end_episode = end_episode + episode_num

    for i_episode in range(episode_num,end_episode):
        print(f"Episode {i_episode} of  {end_episode}")
        observation = env.reset()
        done = False
        score = 0
        count = 0

        camera_name = env.sim.model.camera_names[0]  # Assuming we want the first camera
        frames = []  # Maintain a list of frames for video capture
        while not done:
            count += 1
            action = agent.choose_action(observation, validation=True)
            next_observation, reward, done, info = env.step(action)
            env.render()  # Get rendered frame
            image = env.sim.render(
                camera_name=camera_name,
                width=1024,
                height=768
            )
            flipped_image = np.flipud(image)  # Flip the image vertically
            frames.append(flipped_image)  # Store the flipped frame
            #env.render()
            time.sleep(0.04)
            score += reward
            if count % 50 == 0:
                print(f'{batch_num}\t{count}\trewaward = {reward:.7f}, score = {score:.7f}')
            observation = next_observation
            time.sleep(0.0)

        #if i_episode >= end_episode-2:
        create_video(frames, f'./vids/{ckpt_identifier}_episode_{i_episode}.mp4')  # Save frames as a video


