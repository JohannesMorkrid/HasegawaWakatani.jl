import h5py
def display_h5_contents(file_path):
    """
    Display the contents of an HDF5 (.h5) file, including groups and datasets.
    Parameters:
        file_path (str): Path to the .h5 file.
    """
    def print_structure(name, obj):
        """
        Helper function to print the structure of the HDF5 file.
        """
        if isinstance(obj, h5py.Group):
            print(f"Group: {name}")
        elif isinstance(obj, h5py.Dataset):
            print(f"Dataset: {name}, Shape: {obj.shape}, Data Type: {obj.dtype}")
    try:
        # Open the HDF5 file in read mode
        with h5py.File(file_path, 'r') as h5_file:
            print(f"Contents of file: {file_path}")
            print("-" * 40)
            # Visit all groups and datasets recursively
            h5_file.visititems(print_structure)
    except Exception as e:
        print(f"Error reading the file: {e}")
# Example usage
if __name__ == "__main__":
    # Replace 'example.h5' with the path to your .h5 file
    file_path = "Garcia 2005 PoP.h5"
    display_h5_contents(file_path)
