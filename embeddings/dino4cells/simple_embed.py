import torch
import archs.vision_transformer as vits
import numpy as np

# Check if MPS (Metal Performance Shaders) is available on macOS
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
if (
    device.type == "cpu"
    and hasattr(torch, "mps")
    and hasattr(torch.mps, "is_available")
    and torch.mps.is_available()
):
    print("MPS (Metal) is available! Using Apple GPU acceleration")
    device = torch.device("mps")
else:
    print(f"Using device: {device}")


def dino_model(pretrained_weights_path):
    model = vits.vit_base(img_size=[128], patch_size=16, num_classes=0, in_chans=5)
    embed_dim = model.embed_dim

    model.eval()
    model.to(device)
    print(f"Model loaded to device: {device}")

    state_dict = torch.load(
        pretrained_weights_path, map_location="cpu", weights_only=False
    )
    if "teacher" in state_dict:
        teacher = state_dict["teacher"]
        teacher = {k.replace("module.", ""): v for k, v in teacher.items()}
        teacher = {k.replace("backbone.", ""): v for k, v in teacher.items()}
        msg = model.load_state_dict(teacher, strict=False)
    else:
        assert False, "teacher not in state dict"
        # teacher only for embeddings

    assert len(msg.missing_keys) == 0, print(msg)

    model = model.eval()

    tmp = torch.tensor(
        np.random.uniform(0, 1, (56, 5, 128, 128)).astype(np.float32)
    ).to(device)
    with torch.inference_mode():
        assert model(tmp).shape == (56, embed_dim)

    def eval_network(x):
        with torch.inference_mode():
            x = torch.as_tensor(x)
            return model(x.to(device)).cpu().numpy()

    return eval_network
