import os

file_path = "/path/to/keras_Model.h5"

if os.path.exists(file_path):
    print(f"The file '{file_path}' exists.")
    model = load_model(file_path, compile=False)
else:
    print(f"The file '{file_path}' does not exist.")

