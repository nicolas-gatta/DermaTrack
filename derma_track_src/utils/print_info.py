import torch
import sys

def print_pth_file_contents(pth_path):
    try:
        data = torch.load(pth_path, map_location=torch.device('cpu'))

        if isinstance(data, dict):
            if 'model_state_dict' in data:
                print("The .pth file contains 'model_state_dict':\n")
                state_dict = data['model_state_dict']
                print(f"Type: {type(state_dict)}")
                print("Keys:")
                for key in state_dict:
                    print(f"- {key}")
            else:
                print("The .pth file contains a dictionary with the following keys:\n")
                state_dict = data['state_dict']
                print(f"Type: {type(state_dict)}")
                print("Keys:")
                for key in state_dict:
                    print(f"- {key}")
        else:
            print(f"The .pth file contains an object of type: {type(data)}")
            try:
                print("Trying to access its state_dict:\n")
                print(data.state_dict().keys())
            except Exception as e:
                print("Could not access state_dict:", str(e))

    except Exception as e:
        print("Error loading .pth file:", str(e))


if __name__ == "__main__":
    print_pth_file_contents("C:\\Users\\Utilisateur\\Downloads\\SRResNet_x4-SRGAN_ImageNet.pth.tar")
    print_pth_file_contents("C:\\Users\\Utilisateur\\Desktop\\Projet\\DermaTrack\\derma_track_src\\super_resolution\\models\\SRResNet_x4_BGR2RGB.pth")
