import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image


preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    return preprocess(image).unsqueeze(0)


def extract_patches(image, patch_size):
    _, C, H, W = image.shape
    patches = image.unfold(2, patch_size, 1).unfold(3, patch_size, 1)
    patches = patches.contiguous().view(-1, C, patch_size, patch_size)
    return patches

def compute_lp1(generated, style, patch_size=3):
   
    gen_patches = extract_patches(generated, patch_size)
    style_patches = extract_patches(style, patch_size)

   
    gen_patches = F.normalize(gen_patches.view(gen_patches.size(0), -1), dim=1)
    style_patches = F.normalize(style_patches.view(style_patches.size(0), -1), dim=1)

    similarity = torch.mm(gen_patches, style_patches.T)
    max_similarity = torch.max(similarity, dim=1).values 
    return max_similarity.mean().item()

def compute_lp2(generated, style):

    gen_patches = extract_patches(generated, patch_size=3)
    gen_patches = F.normalize(gen_patches.view(gen_patches.size(0), -1), dim=1)

   
    dot_product = torch.mm(gen_patches, gen_patches.T) 
    diversity = 1 - (dot_product.sum(dim=1) / gen_patches.size(0)) 
    return diversity.mean().item()


def compute_lp(generated, style):

    lp1 = compute_lp1(generated, style, patch_size=3)
    lp2 = compute_lp2(generated, style)
    return (lp1 + lp2) / 2


if __name__ == "__main__":
   
    generated_image_path = "../Project/output/escher.png"
    style_image_path = "../Project/input/escher.jpg"
    
    generated = load_image(generated_image_path)
    style = load_image(style_image_path)

   
    lp_score = compute_lp(generated, style)
    print("Local Patterns Score:", lp_score)
