import torch
from enum import Enum

class ModelFile(str, Enum):
    ARCHITECTURE = "architecture"
    SCALE = "scale"
    MODE = "color_mode"
    INVERT_MODE = "invert_color_mode"
    NEED_RESIZE = "need_resize"
    PATCH_SIZE = "patch_size"
    STRIDE = "stride"
    MULTI = "multi_input"



def add_missing_data(pth_path: str, values: dict) -> None:
    """
    Add the missing data into the models .pth file

    Args:
        pth_path (str): Path to the model
        values (dict): Dict of keys and values to add or update.
    """
    try:
        data = torch.load(pth_path, map_location='cpu', weights_only=True)

        if isinstance(data, dict):
            if 'model_state_dict' in data or 'state_dict' in data:
                modified = False
                for field in ModelFile:
                    if field.value not in data:
                        data[field.value] = values[field]
                        print(f"Added missing field: {field.value} = {values[field]}")
                        modified = True
                    elif data[field.value] != values[field]:
                        data[field.value] = values[field]
                        print(f"Update field: {field.value} = {values[field]}")
                        modified = True
                    else:
                        print(f"Field already have the right value: {field.value} = {data[field.value]}")

                if modified:
                    torch.save(data, pth_path)
                    print(f"Updated .pth file saved: {pth_path}\n")
                else:
                    print("File is up to date.\n")
            else:
               print("The .pth file is not refactor to be use in the software") 

    except Exception as e:
        print("Error loading .pth file:", str(e))


if __name__ == "__main__":
    
    basic_values = {
        ModelFile.ARCHITECTURE: None,
        ModelFile.SCALE: 2,
        ModelFile.MODE: "BGR2RGB",
        ModelFile.INVERT_MODE: "RGB2BGR",
        ModelFile.NEED_RESIZE: False,
        ModelFile.PATCH_SIZE: None,
        ModelFile.STRIDE: None,
        ModelFile.MULTI: False
    }
    
    basic_values[ModelFile.ARCHITECTURE] = "SRCNN"
    basic_values[ModelFile.SCALE] = 2
    basic_values[ModelFile.PATCH_SIZE] = 32
    basic_values[ModelFile.STRIDE] = 16
    basic_values[ModelFile.NEED_RESIZE] = True
    basic_values[ModelFile.MODE]= "BGR2YCrCb"
    basic_values[ModelFile.INVERT_MODE]= "YCrCb2BGR"
    add_missing_data("C:\\Users\\Utilisateur\\Desktop\\Projet\\DermaTrack\\derma_track_src\\super_resolution\\models\\SRCNN_x2_BGR2YCrCb.pth", values = basic_values)
    
    basic_values[ModelFile.ARCHITECTURE] = "SRCNN"
    basic_values[ModelFile.SCALE] = 4
    add_missing_data("C:\\Users\\Utilisateur\\Desktop\\Projet\\DermaTrack\\derma_track_src\\super_resolution\\models\\SRCNN_x4_BGR2YCrCb.pth", values = basic_values)
    
    basic_values[ModelFile.ARCHITECTURE] = "RRDBNet"
    basic_values[ModelFile.SCALE] = 4
    basic_values[ModelFile.PATCH_SIZE] = 128
    basic_values[ModelFile.STRIDE] = None
    basic_values[ModelFile.NEED_RESIZE] = False
    basic_values[ModelFile.MODE] = "BGR2RGB"
    basic_values[ModelFile.INVERT_MODE] = "RGB2BGR"
    add_missing_data("C:\\Users\\Utilisateur\\Desktop\\Projet\\DermaTrack\\derma_track_src\\super_resolution\\models\\Pretrained_RRDB_PSNR_x4_BGR2RGB.pth", values = basic_values)
    
    basic_values[ModelFile.ARCHITECTURE] = "ESRGAN"
    add_missing_data("C:\\Users\\Utilisateur\\Desktop\\Projet\\DermaTrack\\derma_track_src\\super_resolution\\models\\Pretrained_ESRGAN_x4_BGR2RGB.pth", values = basic_values)
    
    basic_values[ModelFile.ARCHITECTURE] = "ESRGAN"
    add_missing_data("C:\\Users\\Utilisateur\\Desktop\\Projet\\DermaTrack\\derma_track_src\\super_resolution\\models\\Medical_ESRGAN_x4_BGR2RGB.pth", values = basic_values)
    
    basic_values[ModelFile.ARCHITECTURE] = "SRResNet"
    basic_values[ModelFile.SCALE] = 2
    basic_values[ModelFile.PATCH_SIZE] = 96
    add_missing_data("C:\\Users\\Utilisateur\\Desktop\\Projet\\DermaTrack\\derma_track_src\\super_resolution\\models\\Pretrained_SRResNet_x2_BGR2RGB.pth", values = basic_values)
    
    basic_values[ModelFile.SCALE] = 4
    add_missing_data("C:\\Users\\Utilisateur\\Desktop\\Projet\\DermaTrack\\derma_track_src\\super_resolution\\models\\Pretrained_SRResNet_x4_BGR2RGB.pth", values = basic_values)
    
    basic_values[ModelFile.ARCHITECTURE] = "SRGAN"
    basic_values[ModelFile.SCALE] = 2
    add_missing_data("C:\\Users\\Utilisateur\\Desktop\\Projet\\DermaTrack\\derma_track_src\\super_resolution\\models\\Pretrained_SRGAN_x2_BGR2RGB.pth", values = basic_values)
    
    basic_values[ModelFile.SCALE] = 4
    add_missing_data("C:\\Users\\Utilisateur\\Desktop\\Projet\\DermaTrack\\derma_track_src\\super_resolution\\models\\Pretrained_SRGAN_x4_BGR2RGB.pth", values = basic_values)
    
    basic_values[ModelFile.ARCHITECTURE] = "EDVR"
    basic_values[ModelFile.PATCH_SIZE] = 256
    basic_values[ModelFile.MULTI] = True
    add_missing_data("C:\\Users\\Utilisateur\\Desktop\\Projet\\DermaTrack\\derma_track_src\\super_resolution\\models\\Pretrained_EDVR_x4_BGR2RGB.pth", values = basic_values)

    add_missing_data("C:\\Users\\Utilisateur\\Desktop\\Projet\\DermaTrack\\derma_track_src\\super_resolution\\models\\Medical_EDVR_x4_BGR2RGB.pth", values = basic_values)
    
    
