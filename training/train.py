####
# Train your model.
#

import json
import sys
import torch

import importlib.util

# Function to load the model
def load_model(model_path, model_name, **kwargs):
    try:
        # Load the module
        spec = importlib.util.spec_from_file_location('model_module', model_path)
        model_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model_module)

        # Import the model class from the module
        Model = getattr(model_module, model_name)

        # Instantiate the model
        model = Model(**kwargs)

        return model
    except Exception as e:
        print("Error loading model:", e)
        return None

# read input as json 
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python3 train.py <params.json>")
        sys.exit(1)
    
    json_file = sys.argv[1]

    # read json params
    try:
        with open(json_file,'r') as f:
            input_json = f.read()
    except FileNotFoundError as e:
        print(f"Error: {e}\nThis error came from train.py. File '{json_file}' not found.")
        sys.exit(1)
    
    # parse json
    try:
        params = json.loads(input_json)
    except json.JSONDecodeError as e:
        print(f"Error: {e}\nThis error came from train.py. Error decoding json from file '{json_file}'.")
        sys.exit(1)
    
    # access params
    model_path = params.get('model_path')
    model_name = params.get('model_name')

    n_embd = params.get('n_embd')
    block_size = params.get('block_size')
    num_heads = params.get('n_heads')
    head_size = params.get('head_size')
    dropout = params.get('dropout')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_layer = params.get('n_layer')
    learning_rate = params.get('learning_rate')
    train_file = params.get("train_file")

    ##
    # TODO:
    # Create function to get vocab size
    # Create function to get unique chars
    # Create function to encode and decode
    # Create function to load data
    # Create function to split data
    #

    kwargs_model = {
        'vocab_size': vocab_size,
        'n_embd':n_embd,
        'block_size':block_size,
        'n_heads':num_heads,
        'head_size':head_size,
        'dropout':dropout,
        'device':device
    }
    print(kwargs_model)

    model = load_model(model_path,model_name,**kwargs_model)
    print(model)