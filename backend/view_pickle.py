import pickle
import pandas as pd

def view_pickle(pickle_path):
    """Load and display contents of a pickle file"""
    try:
        with open(pickle_path, 'rb') as file:
            data = pickle.load(file)

        print(f"\nType of loaded data: {type(data)}")

        # If it's a DataFrame
        if isinstance(data, pd.DataFrame):
            print("\nDataFrame Head:")
            print(data.head())
            print("\nDataFrame Info:")
            print(data.info())

        # If it's a LabelEncoder or similar
        elif hasattr(data, 'classes_'):
            print("\nDetected LabelEncoder or similar.")
            print("Classes:", data.classes_)

        # If it's a OneHotEncoder
        elif hasattr(data, 'get_feature_names_out'):
            print("\nDetected OneHotEncoder or similar.")
            print("Feature Names:", data.get_feature_names_out())

        # General printing for other object types
        else:
            print("\nContent:")
            print(data)

    except Exception as e:
        print(f"Error loading {pickle_path}: {str(e)}")

# Your actual .pkl files
pickle_files = [
    'crop_encoder.pkl',
    'state_encoder.pkl',
    'crop_model.pkl'
]

for pkl_file in pickle_files:
    print(f"\n{'='*50}")
    print(f"Loading: {pkl_file}")
    print(f"{'='*50}")
    view_pickle(pkl_file)
