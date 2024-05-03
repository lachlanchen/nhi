import numpy as np
import argparse
import os

def npy_to_shape(folder_path):
    # Search for the first .npy file in the given folder
    for file in os.listdir(folder_path):
        if file.endswith(".npy"):
            npy_file_path = os.path.join(folder_path, file)
            break
    else:
        print(f"No .npy file found in the folder {folder_path}.")
        return
    
    # Load the frames from the .npy file
    frames = np.load(npy_file_path)
    # Output the shape of the frames
    print(f"Frames shape of {folder_path}: ", frames.shape)

def process_all_folders(base_path, prefix):
    # Process all folders that start with the given prefix
    for folder in os.listdir(base_path):
        if folder.startswith(prefix):
            full_path = os.path.join(base_path, folder)
            if os.path.isdir(full_path):
                npy_to_shape(full_path)

def main():
    parser = argparse.ArgumentParser(description='Get the shape of frames stored in a .npy file.')
    parser.add_argument('folder_name', type=str, nargs='?', default=None, help='Name of the folder containing the .npy file')
    parser.add_argument('-a', '--all', action='store_true', help='Process all folders starting with "data_"')

    args = parser.parse_args()

    if args.all:
        # If the -a option is used, process all folders starting with 'data_'
        base_path = os.getcwd()  # Assuming the script is run from the parent directory
        process_all_folders(base_path, 'data_')
    elif args.folder_name:
        # Process the specified single folder
        npy_to_shape(args.folder_name)
    else:
        print("Please specify a folder name or use the -a option to process all data folders.")

if __name__ == '__main__':
    main()
