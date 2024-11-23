import torch
import torchvision.models as models
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image


vgg = models.vgg19(pretrained=True).features.eval()


def extract_features(image, layers, model):
    features = []
    x = image
    for i, layer in enumerate(model):
        x = layer(x)
        if str(i) in layers:
            features.append(x)
    return features

def cosine_similarity(feature1, feature2):
   
    feature1 = feature1.view(feature1.size(0), -1)
    feature2 = feature2.view(feature2.size(0), -1)

   
    dot_product = (feature1 * feature2).sum(dim=1) 
    norm1 = torch.linalg.norm(feature1, dim=1)
    norm2 = torch.linalg.norm(feature2, dim=1)

    return dot_product / (norm1 * norm2)


def compute_cf(generated, content, layers):
    gen_features = extract_features(generated, layers, vgg)
    content_features = extract_features(content, layers, vgg)
    similarities = [cosine_similarity(g, c) for g, c in zip(gen_features, content_features)]
    return torch.mean(torch.stack(similarities)).item()


preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
])


def load_image(image_path):
    image = Image.open(image_path).convert('RGB') 
    image_tensor = preprocess(image).unsqueeze(0) 
    return image_tensor


layers = ['21', '28']  # conv4_1, conv5_1
cf_score = compute_cf(load_image('../Project/output/escher.png'), load_image('../Project/input/lib.jpg'), layers)
print(cf_score)