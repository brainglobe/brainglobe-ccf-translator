# compile_metadata.py
# This script runs all other scripts in order to compile the metadata
# The order is unimportant other than demba must be first (as it initialises the metadata file)

# Define the paths to the metadata files
demba_metadata_path = "demba_metadata.py"
gubra_metadata_path = "gubra_metadata.py"
princeton_metadata_path = "princeton_metadata.py"


# Function to execute a Python file
def execute_file(file_path):
    with open(file_path, "r") as file:
        exec(file.read())


# Execute the metadata files in order
execute_file(demba_metadata_path)
execute_file(gubra_metadata_path)
execute_file(princeton_metadata_path)
