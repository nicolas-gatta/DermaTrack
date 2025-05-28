import torch

def print_pth_file_contents_dict_state(pth_path):
    try:
        data = torch.load(pth_path, map_location=torch.device('cpu'),  weights_only=True)

        if isinstance(data, dict):
            if 'model_state_dict' in data:
                print("The .pth file contains 'model_state_dict':\n")
                state_dict = data['model_state_dict'].items()
            elif 'state_dict' in data:
                print("The .pth file contains 'state_dict':\n")
                state_dict = data['state_dict'].items()
            else:
                state_dict = data.items()
                
            print(f"Type: {type(state_dict)}")
            print("Keys:")
            for key, value in state_dict:
                if value.shape == torch.Size([1]):
                    print(f"- {key} but problem in the value: {value}")
                else:
                    print(f"- {key}")

    except Exception as e:
        print("Error loading .pth file:", str(e))

def print_pth_file_contents(pth_path):
    try:
        data = torch.load(pth_path, map_location=torch.device('cpu'), weights_only=True)

        if isinstance(data, dict):
            if 'model_state_dict' in data or 'state_dict' in data:
                for key, value in data.items():
                    if key in ["model_state_dict", "state_dict"]:
                        print(f"- {key}: Lot's of info here")
                    else:
                        print(f"- {key}: {value}")
            
            else:
                print("The .pth file is not refactor to be use in the software")

    except Exception as e:
        print("Error loading .pth file:", str(e))
        

if __name__ == "__main__":
    # print_pth_file_contents_dict_state("C:\\Users\\Utilisateur\\Downloads\\RRDB_ESRGAN_x4.pth")
    
    print_pth_file_contents("C:\\Users\\Utilisateur\\Desktop\\Projet\\DermaTrack\\derma_track_src\\super_resolution\\models\\Pretrained_ESRGAN_x4.pth")
