import os
import subprocess
import argparse
import sys

LOCAL_LOG_DIR = "/scratch2/vairaviv/IsaacLab-Internal/logs/"
CLUSTER_LOG_DIR = "/cluster/home/vairaviv/isaaclab/logs/rsl_rl/crowd_navigation/"
CLUSTER_HOME_DIR = "cluster/home/vairaviv"
DOCKER_LOG_DIR = "/workspace/isaaclab/logs/cluster"

def get_running_docker_id():
    """
    Get the ID of the currently running Docker container.
    """
    try:
        # Run `docker ps` to get the list of running containers and their IDs
        result = subprocess.check_output(['docker', 'ps', '-q'], universal_newlines=True)
        # The output will be a list of container IDs, one per line. Return the first one.
        container_id = result.splitlines()[0] if result else None
        if not container_id:
            print("[ERROR] No running Docker containers found.")
            sys.exit(1)
        return container_id
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to get running Docker container ID: {e}")
        sys.exit(1)

def scp_from_cluster(cluster_user, cluster_host, cluster_directory, local_directory):
    """
    Copy files from the cluster to a local directory.
    """
    remote_path = f"{cluster_user}@{cluster_host}:{cluster_directory}"
    try:
        print(f"[INFO] Copying files from {remote_path} to {local_directory}...")
        subprocess.check_call(['scp', '-r', remote_path, local_directory])
        print("[INFO] Files copied to local machine successfully.")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to copy files from cluster: {e}")
        sys.exit(1)

def copy_to_docker(docker_id, local_directory, container_path):
    """
    Copy files from local machine to Docker container and ensure the target directory exists.
    Creates the directory only if it does not exist.
    """
    try:
        # Check if the directory exists in the Docker container
        dir_exists = subprocess.run(
            ['docker', 'exec', docker_id, 'test', '-d', container_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # # If the directory doesn't exist, create it
        # if dir_exists.returncode != 0:  # Directory doesn't exist
        #     print(f"[INFO] Directory {container_path} does not exist in Docker container. Creating it...")
        #     subprocess.check_call(['docker', 'exec', docker_id, 'mkdir', '-p', container_path])
        #     print(f"[INFO] Directory {container_path} created in Docker container.")
        # else:
        #     print(f"[INFO] Directory {container_path} already exists in Docker container.")

        # # Now copy the files from the local directory to Docker container
        # print(container_path)
        print(f"[INFO] Copying files from {local_directory} to Docker container {docker_id}:{container_path}...")
        subprocess.check_call(['docker', 'cp', local_directory, f"{docker_id}:{container_path}"])

        print(f"[INFO] Files copied to Docker container successfully. {local_directory}, {docker_id}:{container_path}")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to copy files to Docker container: {e}")
        sys.exit(1)

def delete_cluster_directory(cluster_user, cluster_host, cluster_directory):
    """
    Delete a directory from the cluster after files are copied.
    """
    remote_path = f"{cluster_user}@{cluster_host}:{cluster_directory}"
    try:
        print(f"[INFO] Deleting directory {cluster_directory} on the cluster...")
        subprocess.check_call(['ssh', f'{cluster_user}@{cluster_host}', f'rm -rf {cluster_directory}'])
        print("[INFO] Directory deleted successfully from the cluster.")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to delete directory from cluster: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Copy files from cluster to local machine, then to a Docker container.")
    
    # Define arguments
    parser.add_argument('--docker_id', default=None, help="Docker container ID. If not specified, it will be auto-detected.")
    parser.add_argument('--directory_name', required=True, help="Directory name on the cluster (e.g., 'logs/'). This will be joined with default paths.")
    parser.add_argument('--cluster_user', default="vairaviv", help="Cluster username.")
    parser.add_argument('--cluster_host', default="euler.ethz.ch", help="Cluster hostname.")
    parser.add_argument('--local_temp_dir', default="./temp_cluster_files", help="Temporary local directory for storing files.")

    args = parser.parse_args()

    # Automatically get the Docker container ID if not provided
    if args.docker_id is None:
        args.docker_id = get_running_docker_id()

    # Combine the paths with the provided directory name
    cluster_directory = os.path.join(CLUSTER_LOG_DIR, args.directory_name)
    # print("the cluster directory is:", cluster_directory)
    local_directory = os.path.join(LOCAL_LOG_DIR, args.directory_name)
    # print("the local directory is:", local_directory)
    # docker_directory = os.path.join(DOCKER_LOG_DIR, args.directory_name)
    # print("the docker directory is:", docker_directory)
    docker_directory = DOCKER_LOG_DIR

    # Copy files from the cluster to a local temporary directory
    # os.makedirs(local_directory, exist_ok=True)
    scp_from_cluster(args.cluster_user, args.cluster_host, cluster_directory, local_directory)

    # Copy the files from the local directory to the specified Docker container path
    copy_to_docker(args.docker_id, local_directory, docker_directory)

    # Delete the directory from the cluster
    delete_cluster_directory(args.cluster_user, args.cluster_host, cluster_directory)

    # # Clean up the temporary local directory
    # try:
    #     subprocess.check_call(['rm', '-rf', args.local_temp_dir])
    #     print(f"[INFO] Temporary directory {args.local_temp_dir} removed.")
    # except Exception as e:
    #     print(f"[WARNING] Could not remove temporary directory: {e}")

if __name__ == "__main__":
    main()
