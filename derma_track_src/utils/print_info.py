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
            print(f"Type: {type(state_dict)}")
            print("Keys:")
            for key, value in state_dict:
                if value.shape == torch.Size([1]):
                    print(f"- {key} but problem in the value: {value}")
                else:
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
    print_pth_file_contents("X")
