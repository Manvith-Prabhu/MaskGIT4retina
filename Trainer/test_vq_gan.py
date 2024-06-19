import torch
import yaml
import torchvision.transforms as transforms
import os
from tqdm import tqdm
from torchvision.datasets import ImageFolder
from omegaconf import OmegaConf
import sys
sys.path.insert(0, '/home/kwang/mprabhu/Modified_MaskGit')
from Network.Taming.models.vqgan import VQModel
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from Metrics.inception_metrics import MultiInceptionMetrics
import numpy as np
from PIL import Image



def remap_image_torch(image):
    min_norm = image.min(-1)[0].min(-1)[0].min(-1)[0].view(-1, 1, 1, 1)
    max_norm = image.max(-1)[0].max(-1)[0].max(-1)[0].view(-1, 1, 1, 1)
    image_torch = ((image - min_norm) / (max_norm - min_norm)) * 255
    image_torch = torch.clip(image_torch, 0, 255).to(torch.uint8)
    return image_torch

def compute_true_features(dataloader, inception=None, max_num=20000):
    if len(dataloader.dataset) < max_num:
        num = len(dataloader.dataset)
    else:
        num = max_num
    bar = tqdm(dataloader, leave=False, desc='computing true images features')
    for i, (images, _) in enumerate(bar):
        if i * dataloader.batch_size >= num:
            break
        images = images.to('cuda')
        inception.update(remap_image_torch(images),image_type="real")

def compute_fake_features(model, dataloader, inception=None, max_num=20000):
    if len(dataloader.dataset) < max_num:
        num = len(dataloader.dataset)
    else:
        num = max_num
    bar = tqdm(dataloader, leave=False, desc="Computing fake images features")
    for i, (images, _) in enumerate(bar):
        if i * dataloader.batch_size >= num:
            break
        images = images.to('cuda')
        with torch.no_grad():
            images, _, _ = model(images)
            images = images.float()
            inception.update(remap_image_torch(images), image_type="unconditional")
torch.cuda.set_device(0)
# load config
vqgan_folder = 'vq_vae'
path = os.path.join(
            '/home/kwang/mprabhu/Modified_MaskGit/pretrained_maskgit',
            vqgan_folder,
            'model_config.yaml'
        )
with open(path, "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
# load inception v3 metrics
inception_metrics = MultiInceptionMetrics(
            reset_real_features=False,
            compute_unconditional_metrics=True,
            compute_conditional_metrics=False,
            compute_conditional_metrics_per_class=False,
            num_classes=config['model']['vq_params']['feature_size'],
            num_inception_chunks=10,
            manifold_k=3,
        )
model = VQModel(vqparams=config['model']['vq_params'], **config['model']['params'])
checkpoint_folder = os.path.join(
            '/home/kwang/mprabhu/Modified_MaskGit/pretrained_maskgit/',
            vqgan_folder,
            'last.ckpt'
        )
checkpoint = torch.load(checkpoint_folder, map_location="cpu")["state_dict"]
# Load network
model.load_state_dict(checkpoint, strict=False)
model = model.eval()
model = model.to("cuda")
transform =transforms.Compose([
                                transforms.Resize(256),
                                transforms.RandomCrop((256, 256)),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize((0.485,0.456,0.406), (0.229, 0.224,0.225))
                            ])
#load dataset
data_test = ImageFolder(root='/home/kwang/mprabhu/Dataset/Preprocessed_ODIR5k/Testing', transform=transform)
test_loader = DataLoader(data_test,
                        batch_size=8,
                        shuffle=True,
                        num_workers=4,
                        pin_memory=True,
                        )
#compute features
compute_true_features(test_loader, inception_metrics )
compute_fake_features(model, test_loader, inception_metrics)
print("here")
results = inception_metrics.compute()
print("\n here2")
metrics = {f"Eval/{k}": v for k, v in results.items()}
print(metrics)

# Output directory for saving images
output_dir = "Recon_images"
os.makedirs(output_dir, exist_ok=True)

# def interpolate(batch):
#     arr = []
#     for img in batch:
#         pil_img = transforms.ToPILImage()(img)
#         resized_img = pil_img.resize((299,299), Image.BILINEAR)
#         arr.append(transforms.ToTensor()(resized_img))
#     return torch.stack(arr)

def _inverse_norm(images):
  if isinstance(images, torch.Tensor):
  # Tensor image to numpy
      images = images.cpu().permute(1, 2, 0).numpy()
      NORM_MEAN = np.array([0.485,0.456,0.406])
      NORM_STD = np.array([0.229, 0.224,0.225])
      images = (images * NORM_STD[None,None]) + NORM_MEAN[None,None]
      images = np.clip(images, a_min=0.0, a_max=1.0)
      images = (images * 255).astype(np.uint8)
  return images

# Iterate over batches
for batch_idx, batch in enumerate(test_loader):
    real_imgs = batch[0].to("cuda")
    reco_imgs, _, perplexity = model(real_imgs)
    print(f'perplexity : {perplexity} | ')
    # Iterate over images in the batch
    for i in range(len(real_imgs)):
        inv_img = _inverse_norm(reco_imgs.detach().cpu()[i])
        # Convert the tensor to a PIL Image
        img = transforms.ToPILImage()(inv_img)
        inv_img = Image.fromarray(inv_img)
        # Save the image
        inv_img_path = os.path.join(output_dir, f"batch_{batch_idx}_image_{i}.png")
        inv_img.save(inv_img_path)
    if batch_idx==5:
        break

print("Images saved successfully.")