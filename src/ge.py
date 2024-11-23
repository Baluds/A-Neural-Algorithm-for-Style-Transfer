import numpy as np
import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image



def compute_color_histogram(image, bins=256):
    if len(image.shape) == 4:
        image = image.squeeze(0)

    if image.shape[0] != 3: 
        raise ValueError("Expected image with 3 color channels, got shape {}".format(image.shape))

   
    image = image.permute(1, 2, 0).numpy()

    histograms = [np.histogram(image[:, :, i], bins=bins, range=(0, 1))[0] for i in range(3)]
    return np.array(histograms)

def extract_features(image, layers, model):
    """
    Extract features from specified layers in a PyTorch model.
    """
    features = {}
    x = image
    for i, layer in enumerate(model):
        x = layer(x)
        if str(i) in layers:
            features[str(i)] = x
    return features

def gram_matrix(features):
    """
    Compute the Gram matrix for a feature map.
    """
    _, C, H, W = features.size()
    features = features.view(C, H * W) 
    return torch.mm(features, features.t()) / (C * H * W)


def compute_gc(generated, style):
    gen_hist = compute_color_histogram(generated)
    style_hist = compute_color_histogram(style)
    return np.mean([np.dot(gen_hist[i], style_hist[i]) / (np.linalg.norm(gen_hist[i]) * np.linalg.norm(style_hist[i])) for i in range(3)])

def compute_ht(generated, style, layers):
    """
    Compute holistic texture similarity (HT) using Gram matrices.
    """
    vgg = models.vgg19(pretrained=True).features.eval()  # Load pre-trained VGG-19
    gen_features = extract_features(generated, layers, vgg)
    style_features = extract_features(style, layers, vgg)
    
    similarities = []
    for layer in layers:
        gen_gram = gram_matrix(gen_features[layer])
        style_gram = gram_matrix(style_features[layer])
       
        dot_product = torch.dot(gen_gram.flatten(), style_gram.flatten())
        norm_gen = torch.norm(gen_gram)
        norm_style = torch.norm(style_gram)
        similarities.append(dot_product / (norm_gen * norm_style))
    return torch.mean(torch.tensor(similarities)).item()

def compute_ge(generated, style, layers):
    gc = compute_gc(generated, style)
    ht = compute_ht(generated, style, layers)
    return (gc + ht) / 2


preprocess = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
])


def load_image(image_path):
    image = Image.open(image_path).convert('RGB') 
    image_tensor = preprocess(image).unsqueeze(0) 
    return image_tensor


style_layers = ['0', '5', '10', '19', '28']  # Conv1_1, Conv2_1, ..., Conv5_1


ge_score = compute_ge(load_image('../Project/output/escher.png'), load_image('../Project/input/escher.jpg'), style_layers)
print("Global Effects Score:", ge_score)
