#tensor_batch.py

'''
The script automates the process of manually launching TensorBoard for each training session.

Input:
batch_training_count.txt: A text file containing the current batch number. This number is used
for the script to identify which training logs to monitor.
Training session log directories: These are located in the specified logs directory (./logs by
default). The script expects these directories to be named following a specific pattern that
includes the batch number.

Output:
TensorBoard Instances: For each relevant training log directory, the script launches a TensorBoard
instance.
Console Messages: The script outputs which TensorBoard instances are being launched and on which
ports they are available.
'''
import subprocess
import os
import sys


def find_free_port():
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


def get_running_tensorboard_dirs():
    # Get the list of running TensorBoard processes
    result = subprocess.run(['pgrep', '-af', 'tensorboard'], stdout=subprocess.PIPE, text=True)
    running_dirs = []

    # Extract the logdir paths from the TensorBoard processes
    for line in result.stdout.splitlines():
        parts = line.split('--logdir ')
        if len(parts) > 1:
            dir_path = parts[1].split()[0]  # Get the path which is immediately after --logdir
            running_dirs.append(dir_path)

    return running_dirs


if __name__ == "__main__":
    print(f'Starting tensor_batch.py')
    log_dir = "./logs"

    current_batch_count = sys.argv[1]
    # Read the current batch count
    #batch_count_file = 'batch_training_count.txt'
    '''
    if os.path.exists(batch_count_file):
        with open(batch_count_file, 'r') as file:
            current_batch_count = file.read().strip()
    else:
        raise FileNotFoundError(f"Could not find {batch_count_file}")
    '''
    running_tensorboard_dirs = get_running_tensorboard_dirs()

    # Find new training session directories
    training_sessions = [d for d in os.listdir(log_dir)
                         if os.path.isdir(os.path.join(log_dir, d)) and
                         d.startswith(f"{current_batch_count}_") and
                         os.path.join(log_dir, d) not in running_tensorboard_dirs]

    # Launch a TensorBoard instance for each new training session
    for session in training_sessions:
        session_log_dir = os.path.join(log_dir, session)
        if session_log_dir not in running_tensorboard_dirs:
            port = find_free_port()
            print(f"Launching TensorBoard for {session} on port {port}")
            subprocess.Popen(['tensorboard', '--logdir', session_log_dir, '--port', str(port)])

    print("All new TensorBoard instances have been launched.")
