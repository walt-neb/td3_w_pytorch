#train_batch.py

'''
train_batch.py: A script to automate the launch of multiple training sessions in parallel.

Functionality:
- The script scans the current directory for hyperparameter files (named 'hyp_*.txt').
- It reads and optionally increments a batch training count from 'batch_training_count.txt'.
- For each hyperparameter file found, it launches a separate training session in a new terminal window.
- Each training session executes 'train.py' with a set number (batch count) and the path to the hyperparameter file.

Inputs:
- Hyperparameter files ('hyp_*.txt') in the current directory.
- A batch count from 'batch_training_count.txt' (if the file doesn't exist, the script starts with count 1).

Outputs:
- Launches training sessions in separate terminal windows, each with its specific set of hyperparameters.
- Updates 'batch_training_count.txt' with the current batch count.
- Console messages indicating the progress of launching training sessions.

Usage:
- The script should be executed without arguments: 'python train_batch.py'
- Ensure 'train.py' and the hyperparameter files ('hyp_*.txt') are in the same directory as this script.
'''

import os
import sys
import subprocess
import glob

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit('Usage: python train_batch.py <batch_num>')

    # Step 1: Find all hyperparameter files in the current directory
    hyperparam_files = glob.glob('hyp_*.txt')
    print(f'Found {len(hyperparam_files)} ')
    if len(hyperparam_files) == 0:
        print("No hyperparameter files found.")
        sys.exit(1)

    # Step 2: Read and increment the batch training count
    batch_num = int(sys.argv[1])

    # Step 3: Launch a training job for each hyperparameter file in a new terminal

    for hyp_file in hyperparam_files:
        print(f'Processing {hyp_file}')
        try:
            base_name = os.path.splitext(os.path.basename(hyp_file))[0]  # Extract the base filename without extension
            window_title = f"Training Session: {base_name}"
            print(f"Launching training session {batch_num} with {hyp_file}")
            terminal_command = f'gnome-terminal --title="{window_title}" -- python train_old.py {str(batch_num)} {hyp_file}'
            subprocess.run(terminal_command, shell=True, check=True)
        except Exception as e:
            print(f'{e}')

    print("All training sessions launched.")

