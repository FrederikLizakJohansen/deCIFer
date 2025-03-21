import pickle
import gzip
import sys

def convert_pkl_to_gz(input_file, output_file):
    """Converts a .pkl file to a .pkl.gz file."""
    try:
        # Read the .pkl file
        with open(input_file, 'rb') as pkl_file:
            data = pickle.load(pkl_file)
        
        # Write the data to a .pkl.gz file
        with gzip.open(output_file, 'wb') as gz_file:
            pickle.dump(data, gz_file, protocol=pickle.HIGHEST_PROTOCOL)
        
        print(f"Conversion successful: {input_file} -> {output_file}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_pkl_to_gz.py <input.pkl> <output.pkl.gz>")
    else:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
        convert_pkl_to_gz(input_file, output_file)
