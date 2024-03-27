import os
import json
import hashlib
from abc import ABC, abstractmethod


class MetadataLogger(ABC):

    def __init__(self, metadata_folder, args):
        """
        Initializer for the MetadataLogger.

        Parameters:
        - metadata_folder: str, the path to the folder where metadata files are stored.
        """
        self.metadata_folder = metadata_folder
        os.makedirs(self.metadata_folder, exist_ok=True)
        self.inputs = {'metadata_folder': metadata_folder,
                       'args': args}
        self.outputs = {}

    def generate_metadata_file_path(self):
        """Generate a unique metadata file path based on the inputs."""
        inputs_str = str(self.inputs)
        inputs_hash = hashlib.sha256(inputs_str.encode()).hexdigest()
        return os.path.join(self.metadata_folder, inputs_hash + ".json")

    def save_metadata(self):
        """Save the current state of inputs and outputs to a metadata file."""
        metadata = {"inputs": self.inputs, "outputs": self.outputs}
        with open(self.generate_metadata_file_path(), "w") as f:
            json.dump(metadata, f)

    def load_metadata(self):
        """Load the saved metadata for the current inputs."""
        metadata_file_path = self.generate_metadata_file_path()
        if os.path.exists(metadata_file_path):
            with open(metadata_file_path, "r") as f:
                return json.load(f)
        return None

    def outputs_exist(self):
        """
        Check if previous outputs for the current inputs exist and are valid.

        Returns:
        - True if all outputs exist, False otherwise.
        """
        metadata = self.load_metadata()
        if metadata:
            for output_path, data_label in metadata["outputs"].items():
                if not os.path.exists(output_path):
                    return False
            return True
        return False

    @abstractmethod
    def process(self):
        """
        Abstract method that needs to be implemented in the derived class.
        This method will contain the main processing logic.
        """
        pass
