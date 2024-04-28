

#filename: train.py
'''
This project is focused on training and testing a robotic arm's performance using a
robosuite and a robust reinforcement learning model known TD3.
Twin Delayed Deep Deterministic Policy Gradient (TD3) is an advanced actor-critic method
designed to address some challenges in Deep Deterministic Policy Gradient (DDPG),
particularly regarding overestimation bias and policy variance.
'''

import os
import sys
import datetime
from torch.utils.tensorboard import SummaryWriter
import robosuite as suite
from robosuite.wrappers.gym_wrapper import GymWrapper
from td3_torch import Agent_TD3
import psutil
process = psutil.Process(os.getpid())
import buffer
import csv

#from training_helpers import (load_hyperparams, print_parameters, log_metrics)

import cv2
import select
import tty
import termios

def create_video(frames, filename, fps=30):
    height, width, _ = frames[0].shape
    size = (width, height)
    out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)  # Using mp4v codec
    for frame in frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))  # Write out frame to video
    out.release()
    print('Video saved to {}'.format(filename))

def read_keyboard_input():
    # Check if there is input available
    if select.select([sys.stdin], [], [], 0)[0]:
        return sys.stdin.read(1)
    else:
        return None
# def read_keyboard_input(): #(without pressing enter)
#     # Save the terminal settings
#     old_settings = termios.tcgetattr(sys.stdin)
#     try:
#         # Change the terminal to raw mode
#         tty.setraw(sys.stdin.fileno())
#         # Check if input is available
#         if select.select([sys.stdin], [], [], 0)[0]:
#             # Read a single character
#             return sys.stdin.read(1)
#     finally:
#         # Restore the terminal settings
#         termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
#     return None

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


def log_metrics(writer, metrics, i_episode):
    # Log score on a separate graph
    writer.add_scalar('Performance/score', metrics['score'], i_episode)

    # Log Q values on the same graph
    writer.add_scalars('Performance/Q_values_mean', {
        'q1': metrics['mean_q1'],
        'q2': metrics['mean_q2'],
        'next_q1': metrics['mean_next_q1'],
        'next_q2': metrics['mean_next_q2']
    }, i_episode)

    # Log actor and critic loss on separate graphs
    if metrics['actor_loss'] is not None:
        writer.add_scalar('Loss/actor', metrics['actor_loss'], i_episode)
    if metrics['critic_loss'] is not None:
        writer.add_scalar('Loss/critic', metrics['critic_loss'], i_episode)


def log_memory_usage(step):
    mem = process.memory_info().rss / float(2 ** 20)  # Convert to MB
    log_writer.add_scalar('Memory usage (MB)', mem, global_step=step)


# Define a function to log observations into the DataFrame
def log_env_info_to_txt(step_count, observation, action, reward, act_file, obs_file):

    # making sure that sequences are not None and are in list form :
    action = list(action) if action is not None else []
    observation = list(observation) if observation is not None else []

    with open(obs_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([step_count, reward] + observation)

    with open(act_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([step_count] + action)


def print_network_statistics(network, network_name):
    for name, param in network.named_parameters():
        # Assuming the parameters are on CPU for printing
        param = param.data.cpu().numpy()
        print(f"{network_name} - {name}: mean={param.mean()}, std={param.std()}, max={param.max()}, min={param.min()}")


def print_checkpoint_statistics(actor, critics):
    print_network_statistics(actor, "Actor")
    for i, critic in enumerate(critics, start=1):
        print_network_statistics(critic, f"Critic {i}")



def validate_statistics(new_stats, saved_stats, threshold=1e-6):
    error_count = 0
    for key in new_stats:
        for stat in ['mean', 'std', 'max', 'min']:
            diff = abs(new_stats[key][stat] - saved_stats[key][stat])
            if diff > threshold:
                print(f"Warning: {key} {stat} variation {diff} exceeds threshold {threshold}")
                error_count += 1
    return error_count

def load_and_compare_network_statistics(network, network_name, filename, threshold=1e-6):
    new_stats = {}
    for name, param in network.named_parameters():
        param = param.data.cpu().numpy()
        new_stats[f"{network_name}_{name}"] = {
            'mean': param.mean(),
            'std': param.std(),
            'max': param.max(),
            'min': param.min()
        }

    saved_stats = {}
    try:
        filename = filename
        with open(filename, 'r') as file:
            for line in file:
                stat_name, values = line.split(' - ')
                mean, std, max_val, min_val = values.strip().split(', ')
                saved_stats[stat_name] = {
                    'mean': float(mean.split(': ')[1]),
                    'std': float(std.split(': ')[1]),
                    'max': float(max_val.split(': ')[1]),
                    'min': float(min_val.split(': ')[1])
                }
    except Exception as e:
        print(f'{e} \nload_and_compare_network_statistics Failed to load', filename)
        return

    validation_errs = validate_statistics(new_stats, saved_stats, threshold)
    if validation_errs > 0:
        print(f'Checkpoint Validation ERROR for: {network_name}')
    else: print(f'{network_name} network loaded and validated')

def load_and_compare_checkpoint_statistics(agent, checkpoint_file, stats_file, threshold=1e-6):
    print("loading and comparing checkpoint statistics for ", checkpoint_file)
    load_and_compare_network_statistics(agent.actor, "Actor", stats_file+'_actor', threshold)
    load_and_compare_network_statistics(agent.critic_1, "Critic1", stats_file+'_critic1', threshold)
    load_and_compare_network_statistics(agent.critic_2, "Critic2", stats_file+'_critic2', threshold)


if __name__ == "__main__":

    # Obtain hyperparameters from the command line arguments
    if len(sys.argv) > 4 or len(sys.argv) < 3:
        print(f'Usage: python train.py [batch_num] [hyp_file] <checkpoint_input_file>')
        sys.exit(0)
    if len(sys.argv) == 4:
        ckpt_input_file = sys.argv[3]
        ckpt_input_root = ckpt_input_file.rstrip('.ckpt')
    else:
        ckpt_input_file = None
        ckpt_input_root = None
    print(f'sys.argv={sys.argv}')
    batch_num = sys.argv[1]
    hyp_file = sys.argv[2]
    hyp_file_root = hyp_file.rstrip('.txt')

    print(f'batch_num: {batch_num}, hyp_file: {hyp_file}, hyp_file_root: {hyp_file_root}')

    start_time = datetime.datetime.now()
    print(f'train.py Starting training at: {start_time.strftime("%Y-%m-%d  %H:%M:%S")}')

    if not os.path.exists('./logs'):
        os.makedirs('./logs', exist_ok=True)
    checkpoint_dir = './checkpoints'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)

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
        has_renderer=False,
        use_camera_obs=False,
        horizon=params["horizon"],
        reward_shaping=params["reward_shaping"],
        control_freq=params["control_freq"],
    )

    env = GymWrapper(env)
    input_dims = tuple(env.observation_space.shape)
    n_actions = env.action_space.shape[0]
    print(f'input_dims: {input_dims}, n_actions: {n_actions}')

    try:
        layer_dims_string = params["layer_dims"].strip('[]')
        layer_dims = [int(dim) for dim in layer_dims_string.split()]
    except Exception as e:
        print(e)
    print(f'layer_dims: {layer_dims}')

    agent = Agent_TD3(actor_learning_rate=params["actor_learning_rate"],
                        critic_learning_rate=params["critic_learning_rate"],
                        tau=params["tau"],
                        input_dims=tuple(env.observation_space.shape),
                        env=env,
                        n_actions=env.action_space.shape[0],
                        layer_dims = layer_dims,
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
                          f'l1_{layer_dims}_'                          
                          f'b_{params["batch_size"]}')
    ckpt_identifier = (f'{batch_num}_'
                       f'{hyp_file_root}')

    log_path = './logs'+f'/{batch_num}_{hyp_file_root}' #{training_identifier}'
    log_writer = SummaryWriter(log_path)
    cp_hyp2log = f'cp {hyp_file} {log_path}'
    os.system(cp_hyp2log) #save a copy of the hyperparameters used for training, to the log file

    ckpt_file_name = os.path.join('./checkpoints', ckpt_identifier + '.ckpt')
    replaybuff_file_name = os.path.join('./checkpoints', ckpt_identifier + '.pkl')

    # Define your parameters
    replay_buffer_capacity = params["max_replay_buffer_size"]  # maximum size for the replay buffer
    input_shape = (4,)  # shape of state-space (tuple of integers)
    n_actions = params["n_actions"]  # number of actions possible in your environment

    # Instantiate ReplayBuffer
    replay_buffer = buffer.ReplayBuffer(replay_buffer_capacity, input_shape, n_actions)  # create a replay buffer

    # Load buffer if available
    if os.path.isfile(replaybuff_file_name):
        replay_buffer = buffer.load_replaybuffer(replaybuff_file_name)
        if replay_buffer is None:
            replay_buffer = buffer.ReplayBuffer(replay_buffer_capacity, input_shape, n_actions)
    else:
        print(f'Replay buffer not found, creating new one {replaybuff_file_name}')

    if ckpt_input_file != None:
        start_episode = agent.load_checkpoint(ckpt_input_file)
        stats_file = f'{ckpt_input_file}.stats'
    else:
        start_episode = agent.load_checkpoint(ckpt_file_name)
        stats_file = f'{ckpt_file_name}.stats'

    load_and_compare_checkpoint_statistics(agent, ckpt_file_name, stats_file)
    #start_episode = agent.load_checkpoint(ckpt_file_name)  # loading the model checkpoint
    #print_checkpoint_statistics(agent.actor, [agent.critic_1, agent.critic_2])
    best_score = [0, 0]
    ave_score = 0
    score_a = .92
    score_b = 1-score_a
    last_checkpoint_episode = 2

    # Define file names
    observation_file = f'{batch_num}_{hyp_file_root}_observations.txt'
    action_file = f'{batch_num}_{hyp_file_root}_actions.txt'

    if start_episode is None:
        start_episode = 1
    if start_episode > 60000:
        start_episode = 1
    for i_episode in range(start_episode, params["end_episode"]):
        key_press = read_keyboard_input()

        result = env.reset()
        if isinstance(result, tuple) and len(result) == 2:
            observation, info = result # For robosuite version 1.4.1
        else:
            observation = result # For robosuite version 1.4.0
            info = None
        done = False
        if params.get("log_actions_and_obs") == 'True':
            log_env_info_to_txt(0, observation, None, None, action_file, observation_file)

        score = 0
        step_count = 1  # Start at 1 as we've already logged the initial observation
        while not done:
            action = agent.choose_action(observation)
            next_observation, reward, done, info, temp = env.step(action)

            # Log the details after each step
            if params.get("log_actions_and_obs") == 'True':
                log_env_info_to_txt(step_count, observation, action, reward, action_file, observation_file)
            step_count += 1
            score += reward
            agent.remmember(observation, action, reward, next_observation, done)
            observation = next_observation

            metrics = agent.learn()

            if metrics is not None:
                metrics['score'] = score
            if metrics is not None and 'mean_q1' in metrics and 'mean_q2' in metrics and \
                    'mean_next_q1' in metrics and 'mean_next_q2' in metrics:
                log_metrics(log_writer, metrics, i_episode)
        ave_score = score_a*ave_score + score_b*score
        print(f'{batch_num} {env_name} {hyp_file_root} {i_episode}\t score = {score:.5f} '
              f'best:{best_score[1]:.5f} i_best:{best_score[0]} ave_score:{ave_score:.5f} patience:{training_patience_threshold}')
        log_writer.add_scalar(f'{batch_num}_{hyp_file_root}', score, global_step=i_episode)

        if score > best_score[1]:
            best_score = [i_episode, score]
        if (best_score[0] == i_episode) or (key_press == 'c'):
            training_patience_threshold = params["training_patience_threshold"]
            if i_episode > last_checkpoint_episode + params["min_checkpoint_span"]:
                agent.save_checkpoint(ckpt_file_name+'_best', i_episode)
                buffer.save_replaybuffer(replay_buffer, replaybuff_file_name)
                ckpt_time = datetime.datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
                last_checkpoint_episode = i_episode
                print(f'checkpoint at {ckpt_time}')
        else: training_patience_threshold -= 1

        if training_patience_threshold < 1:
            early_stop = True
            sys.exit(f'Early stopping reached. Last checkpoint: {ckpt_file_name}_last')
            elapsed_time = end_time - start_time
            print(f'Started training on {start_episode} at: \t{start_time.strftime("%Y-%m-%d  %H:%M:%S")}')
            print(f'Ended training on {i_episode} at: \t{end_time.strftime("%Y-%m-%d  %H:%M:%S")}')
            print(f'Total training time:  {str(elapsed_time)}')

        log_memory_usage(i_episode)

    end_time = datetime.datetime.now()
    elapsed_time = end_time - start_time
    agent.save_checkpoint(ckpt_file_name + '_last', i_episode)
    buffer.save_replaybuffer(replay_buffer, replaybuff_file_name)
    ckpt_time = datetime.datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    last_checkpoint_episode = i_episode
    print(f'checkpoint at {ckpt_time}')
    print(f'Started training on {start_episode} at: \t{start_time.strftime("%Y-%m-%d  %H:%M:%S")}')
    print(f'Ended training on {i_episode} at: \t{end_time.strftime("%Y-%m-%d  %H:%M:%S")}')
    print(f'Total training time:  {str(elapsed_time)}')
    print(f'Best Score: {best_score[1]}')


