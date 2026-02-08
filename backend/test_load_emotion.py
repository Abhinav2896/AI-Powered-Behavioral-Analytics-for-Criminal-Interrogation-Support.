
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import sys

# Add backend to path
sys.path.append(os.path.join(os.getcwd(), 'backend'))

model_path = "backend/model_file.h5"

print(f"Testing model loading from: {os.path.abspath(model_path)}")

# Define patches
from tensorflow.keras.layers import InputLayer

class PatchedInputLayer(InputLayer):
    def __init__(self, batch_shape=None, **kwargs):
        if batch_shape is not None:
            kwargs['batch_input_shape'] = batch_shape
        super().__init__(**kwargs)

# Mock DTypePolicy
class DTypePolicy:
    def __init__(self, config=None, **kwargs):
        self.name = "float32"
        if config and isinstance(config, dict) and 'name' in config:
            self.name = config['name']
        self.compute_dtype = self.name
        self.variable_dtype = self.name
    
    @classmethod
    def from_config(cls, config):
        return cls(config)
    
    def get_config(self):
        return {"name": self.name}

custom_objects = {
    'InputLayer': PatchedInputLayer,
    'DTypePolicy': DTypePolicy
}

try:
    print("Attempting load with custom objects...")
    model = load_model(model_path, compile=False, custom_objects=custom_objects)
    print("Load SUCCESS with custom objects!")
except Exception as e:
    print(f"Load FAILED: {e}")
    import traceback
    traceback.print_exc()
