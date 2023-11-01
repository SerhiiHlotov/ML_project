import os
import logging

# Configuration
log_folder = "logs"  # Folder where log files will be saved
log_filename = "log"  # Base name for log files (e.g., log_1.txt, log_2.txt, ...)
max_log_files = 10  # Maximum number of log files to keep

# Create the log folder if it doesn't exist
if not os.path.exists(log_folder):
    os.makedirs(log_folder)

# Get a list of existing log files in the folder
existing_log_files = [file for file in os.listdir(log_folder) if file.startswith(log_filename)]
print(existing_log_files)
# If there are more than 'max_log_files' log files, remove the oldest one
if len(existing_log_files) >= max_log_files:
    oldest_log_file = os.path.join(log_folder, min(existing_log_files))
    os.remove(oldest_log_file)

# Define the log file path
log_file_path = os.path.join(log_folder, f"{log_filename}_{len(existing_log_files) + 1}.log")

# Configure the logger to write log messages to the log file
logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(asctime)s - %(message)s')

# Log a message (replace this with your program code)
logging.info("This is the log message.")

# Continue with your program
