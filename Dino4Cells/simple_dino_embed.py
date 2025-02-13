import torch
import archs.vision_transformer as vits
import numpy as np

pretrained_weights = "/Users/thouis/Desktop/Cell_Painting_data/DINO_cell_painting_base_checkpoint.pth"

def dino_model():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = vits.vit_base(
                    img_size=[128],
                    patch_size=16,
                    num_classes=0,
                    in_chans=5
                )
    embed_dim = model.embed_dim

    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.to(device)

    state_dict = torch.load(pretrained_weights, map_location="cpu")
    if "teacher" in state_dict:
        teacher = state_dict["teacher"]
        teacher = {k.replace("module.", ""): v for k, v in teacher.items()}
        teacher = {
            k.replace("backbone.", ""): v for k, v in teacher.items()
        }
        msg = model.load_state_dict(teacher, strict=False)
    else:
        assert False, "teacher not in state dict"
        # teacher only for embeddings

    for p in model.parameters():
        p.requires_grad = False
    model = model.eval()

    tmp = torch.tensor(np.random.uniform(0, 1, (56, 5, 128, 128)).astype(np.float32))
    assert model(tmp).shape == (56, embed_dim)

    return lambda x: model(torch.tensor(x).to(device)).numpy()

