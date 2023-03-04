import io
import os
from sys import argv

from PIL import Image
import numpy as np
from inference import model_fn, input_fn, predict_fn, output_fn


def test():
    img_path = argv[1]
    img = Image.open(img_path)
    buff = io.BytesIO()
    img.save(buff, 'png')

    model = model_fn('../..')
    input_data = input_fn(buff.getvalue(), 'image/png')
    output = predict_fn(input_data, model)

    output = output.cpu().data.numpy().astype(np.uint8)
    Image.fromarray(output).show()
    # output = output_fn(output, 'application/json')
    # Image.fromarray(np.array(output).astype(np.uint8)).show()


if __name__ == '__main__':
    test()