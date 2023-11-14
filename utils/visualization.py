import torch
from PIL import Image
import os
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def save_images(img_tensors, img_names, save_dir):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    for img_tensor, img_name in zip(img_tensors, img_names):
        tensor = (img_tensor.clone()+1)*0.5 * 255
        tensor = tensor.cpu().clamp(0,255)

        array = tensor.detach().numpy().astype('uint8')
        if array.shape[0] == 1:
            array = array.squeeze(0)
        elif array.shape[0] == 3:
            array = array.swapaxes(0, 1).swapaxes(1, 2)
            
        Image.fromarray(array).save(os.path.join(save_dir, img_name))
    
def save_checkpoint(model, save_path):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    torch.save(model.cpu().state_dict(), save_path)
    model.to(device)

def load_checkpoint(model, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        print('----No checkpoints at given path----')
        return
    model.load_state_dict(torch.load(checkpoint_path))
    model.to(device)
    print('----checkpoints loaded from path: {}----'.format(checkpoint_path))
