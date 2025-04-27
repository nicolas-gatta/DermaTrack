import torch

def print_pth_file_contents(pth_path):
    try:
        data = torch.load(pth_path, map_location=torch.device('cpu'))

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


if __name__ == "__main__":
    print_pth_file_contents("C:\\Users\\Utilisateur\\Downloads\\RRDB_ESRGAN_x4.pth")
