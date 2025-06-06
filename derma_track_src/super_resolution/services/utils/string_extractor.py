
def extract_dataset_name(path: str) -> str:
    """
    Extract the dataset name without the unecessary information

    Args:
        path (str): The path to the dataset

    Returns:
        str: The dataset name
    """
    return path.rsplit("\\", 1)[-1].split("_HR",)[0]+"_HR"