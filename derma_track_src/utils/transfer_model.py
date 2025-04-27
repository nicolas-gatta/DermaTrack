import os, sys

sys.path.insert(1, "\\".join(os.path.realpath(__file__).split("\\")[0:-2]))

from collections import OrderedDict
from super_resolution.services.SRCNN.model import SRCNN
from super_resolution.services.RRDBNet.model import RRDBNet
from super_resolution.services.SRResNet.model import SRResNet
import torch


def get_layers(architecture, scale: int = None):
     model_layers = []
     model = None
     
     match(architecture):
     
          case "SRCNN":
               model =  SRCNN().state_dict() 
          
          case "SRGAN" | "SRResNet":
               model = SRResNet(up_scale = scale).state_dict()
          
          case "ESRGAN" | "RRDBNet":
               model = RRDBNet(up_scale=scale).state_dict()
          
          case _:
               raise ValueError(f"Unknown architecture: {architecture}")
          
     for keys, _ in model.items():
          model_layers.append(keys)
          
     return model_layers

def verify_model(architecture, model_path):
     
     model_info = torch.load(model_path, weights_only=True)
     
     model = None
     
     match(architecture):
     
          case "SRCNN":
               model = SRCNN()
          
          case "SRGAN" | "SRResNet":
               model = SRResNet(up_scale = model_info["scale"])
          
          case "ESRGAN" | "RRDBNet":
               model = RRDBNet(up_scale = model_info["scale"])
          
          case _:
               raise ValueError(f"Unknown architecture: {architecture}")
          
     try:
          model.load_state_dict(model_info["model_state_dict"])
          return True
     except:
          return False

def main(inputs_model_path, output_model_path, architecture, scale, mode, invert_mode, patch_size, stride):
     
     print(f"Strating Transfer of the model {architecture} !")
     model_layers = get_layers(architecture = architecture, scale = scale)
     
     # Process parameter dictionary
     state_dict = torch.load(inputs_model_path, map_location=torch.device("cpu"), weights_only=True)["state_dict"]
     new_state_dict = OrderedDict()

     for k, v in state_dict.items():
          new_state_dict[model_layers.pop(0)] = v

     torch.save({"architecture": architecture, "scale": scale, "color_mode": mode, "invert_color_mode": invert_mode, "need_resize": False, 
                "patch_size": patch_size, "stride": stride, "model_state_dict": new_state_dict}, output_model_path)
     
     if not verify_model(architecture=architecture, model_path=output_model_path):
          raise ValueError(f"Problem append during the loading of the model !")
     
     print(f"End of the transfer for {architecture}, everything went smoothly !")

if __name__ == "__main__":
     main("X", "Y", architecture = "SRGAN", scale = 2, mode = "BGR2RGB", invert_mode = "RGB2BGR", patch_size = None, stride = None)