import torch

DEVICE = torch.device('mps')
SIZE = 512
EPOCHS = 300
STYLE_PATH = '../Project/input/escher.jpg'
STYLE_WEIGHT = 10000
CONTENT_PATH = '../Project/input/lib.jpg'
CONTENT_WEIGHT = 1
OUTPUT_PATH = '../Project/output/escher.png'
BLANK_PATH = '../Project/input/blank.png'