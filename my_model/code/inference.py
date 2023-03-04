import os
import io
import json

import numpy as np
import torch
from PIL import Image
from model import ISNet


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
image_size = 1024


def model_fn(model_dir):
    model_path = os.path.join(model_dir, 'my_model', 'isnet-general-use.pth')
    net = ISNet()
    net.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    net.eval()
    net.to(device)
    return net


def input_fn(request_body, request_content_type):
    assert request_content_type in ('image/jpeg', 'image/png')
    img = Image.open(io.BytesIO(request_body))
    im0 = np.array(img)
    img = img.resize((image_size, image_size))
    img = (np.array(img).astype(np.float32) / 255. - 0.5) / 1.0
    data = np.expand_dims(np.transpose(img, (2, 0, 1)), 0)
    return {
        'input': torch.tensor(data, dtype=torch.float32, device=device),
        'im0': torch.tensor(im0, dtype=torch.uint8, device=device)
    }


def predict_fn(input_data, model):
    inputs = input_data['input']
    im0 = input_data['im0']
    im0 = im0.cpu().data.numpy()
    im0 = Image.fromarray(im0)
    h0, w0 = im0.size
    pred = model(inputs)
    pred = pred.cpu().data.numpy().squeeze()

    mi = np.min(pred)
    ma = np.max(pred)
    pred = (pred - mi) / (ma - mi)
    pred = np.clip(pred * 255, 0, 255).astype(np.uint8)
    pred = Image.fromarray(pred).resize((h0, w0))

    empty = Image.new('RGBA', im0.size, 0)
    composite = Image.composite(im0, empty, pred)
    composite = np.array(composite).astype(np.uint8)
    return composite


def output_fn(predictions, content_type):
    assert content_type == "image/png"
    buff = io.BytesIO()
    image = Image.fromarray(predictions).convert('RGBA')
    image.save(buff, 'png')
    return image.getvalue()