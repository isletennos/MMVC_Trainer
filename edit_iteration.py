import os
import torch

def edit_iteration(checkpoint_path, new_iteration):
    assert os.path.isfile(checkpoint_path), f"No such file or directory: {checkpoint_path}"
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model = checkpoint_dict['model']
    iteration = checkpoint_dict['iteration']
    optimizer = checkpoint_dict['optimizer']
    learning_rate = checkpoint_dict['learning_rate']

    torch.save({'model': model,
                'iteration': new_iteration,
                'optimizer': optimizer,
                'learning_rate': learning_rate},
                checkpoint_path)

edit_iteration("./logs/20220826_jvs_flow8_da_mel/G_50000.pth", 50000)
edit_iteration("./logs/20220826_jvs_flow8_da_mel/D_50000.pth", 50000)
