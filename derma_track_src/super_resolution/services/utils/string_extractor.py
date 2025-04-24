
def extract_dataset_name(path: str):
    return path.rsplit("\\", 1)[-1].split("_HR",)[0]+"_HR"