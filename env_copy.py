from azureml.core import Workspace, Environment

# Connect to the source Azure ML workspace
source_ws = Workspace.from_config()  # Ensure your config.json points to the correct workspace

# Get the original environment from the source workspace
original_env = Environment.get(workspace=source_ws, name="training-env:3")

# Save the environment to a directory
original_env.save_to_directory(path="./env", overwrite=True)

# Connect to the target Azure ML workspace
target_ws = Workspace.from_config(path="./config_target.json")  # Use a separate config file for the target workspace

# Load the environment from the saved directory
new_env = Environment.load_from_directory("./env")

# Register the environment in the target workspace
new_env.register(workspace=target_ws)
print("Environment successfully copied and registered in the target workspace!")