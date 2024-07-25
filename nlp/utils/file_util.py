import os

def string_to_file(text:str, file_path:str) -> None:
    with open(file_path, "w") as file:
        file.write(text)

def check_directory(directory: str) -> None:
    if not os.path.exists(directory):
        os.makedirs(directory)

def check_file_exists(file_path) -> bool:
    return True if os.path.exists(file_path) else False