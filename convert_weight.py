# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Flatten, Dropout, Input

# # Load the weights from the .npy file



# # Load the weights from the .npy file
# weights_path = "eye-condition-detection-deep-learning-main/weights/bottleneck_features_validation.npy"  # Update this path
# weights = np.load(weights_path, allow_pickle=True)

# # Print the shape of each weight array
# for idx, weight in enumerate(weights):
#     print(f"Weights {idx}: shape {weight.shape}")


# # Create the model architecture (ensure this matches your original model)
# model = Sequential()
# model.add(Input(shape=(7, 7, 512)))  # Adjust the input shape as per your model
# model.add(Flatten())
# model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.65))
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.15))
# model.add(Dense(1, activation='sigmoid'))

# # Set the model weights
# model.set_weights(weights)

# # Save the model weights as .h5
# model.save_weights('converted_weights.h5')

import numpy as np
import h5py
import os
import sys
from pathlib import Path

def convert_npy_to_h5(npy_path, h5_path, dataset_name='weights'):
    """
    Convert weights from .npy format to .h5 format with error handling
    
    Parameters:
    npy_path (str): Path to the input .npy file
    h5_path (str): Path where the .h5 file will be saved
    dataset_name (str): Name of the dataset within the H5 file
    """
    try:
        # Convert paths to Path objects for better handling
        npy_path = Path(npy_path)
        h5_path = Path(h5_path)
        
        # Check if input file exists
        if not npy_path.exists():
            raise FileNotFoundError(f"Input file not found: {npy_path}")
            
        # Create output directory if it doesn't exist
        h5_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Check if we have write permission in the output directory
        if not os.access(h5_path.parent, os.W_OK):
            raise PermissionError(f"No write permission in directory: {h5_path.parent}")
        
        # Load the numpy array
        print(f"Loading weights from {npy_path}")
        weights = np.load(npy_path)
        
        # Create an H5 file and save the weights
        print(f"Saving weights to {h5_path}")
        with h5py.File(str(h5_path), 'w') as f:
            # Create a dataset with the weights
            f.create_dataset(dataset_name, data=weights)
            
            # Add some metadata
            f.attrs['original_file'] = str(npy_path)
            f.attrs['conversion_type'] = 'npy_to_h5'
        
        print("Conversion completed successfully!")
        return True
        
    except PermissionError as e:
        print(f"Permission Error: {e}")
        print("Please check that you have write permissions in the output directory")
        print("Try running the script with administrator privileges or choose a different output location")
        return False
    except FileNotFoundError as e:
        print(f"File Error: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        print(f"Error type: {type(e).__name__}")
        return False

if __name__ == "__main__":
    # Get current working directory
    current_dir = Path.cwd()
    
    # Define paths relative to current directory
    npy_file = current_dir / "eye-condition-detection-deep-learning-main/weights/bottleneck_features_train.npy"
    h5_file = current_dir / "C:/Users/HP/OneDrive/Documents/weights"
    
    print(f"Current working directory: {current_dir}")
    print(f"Input file path: {npy_file}")
    print(f"Output file path: {h5_file}")
    
    # Convert the weights
    success = convert_npy_to_h5(npy_file, h5_file)
    
    if success:
        # Verify the conversion
        try:
            with h5py.File(h5_file, 'r') as f:
                print(f"Available datasets: {list(f.keys())}")
                print(f"Shape of weights: {f['weights'].shape}")
        except Exception as e:
            print(f"Error verifying conversion: {e}")