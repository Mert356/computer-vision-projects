import torch
from PIL import Image

from dataset import eval_transform


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')

    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


# Returns the sigmoid dog score between 0 and 1
def predict_image(model, path, device):
    model.eval()
    with Image.open(path) as image:
        tensor = eval_transform(image.convert('RGB'))

    tensor = tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        logit = model(tensor)
        return torch.sigmoid(logit).item()
