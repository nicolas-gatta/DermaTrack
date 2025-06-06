import os

def get_unique_filename(model_name: str, output_path: str) -> str:
    """
    Generates a unique filename by removing the extension and appending a numeric suffix if needed.

    Args:
        model_name (str): The original filename.
        output_path (str): The patht to the folder we want to save our file.

    Returns:
        str: A unique filename without an extension.
    """

    base_name, ext = os.path.splitext(model_name)
    
    existing_names = {os.path.splitext(file)[0] for file in os.listdir(output_path)}
    
    if base_name in existing_names:
        suffix = 1
        new_name = f"{base_name}_{suffix}"
        
        while new_name in existing_names:
            suffix += 1
            new_name = f"{base_name}_{suffix}"
        
        base_name = new_name
    
    return f"{base_name}{ext}"
    