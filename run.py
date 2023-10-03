from transformers import VitMatteImageProcessor
from model.pt_vitmatte import VitMatteForImageMatting
import torch
from PIL import Image
from huggingface_hub import hf_hub_download

processor = VitMatteImageProcessor.from_pretrained(
    "hustvl/vitmatte-small-composition-1k"
)
model = VitMatteForImageMatting.from_pretrained("hustvl/vitmatte-small-composition-1k")

filepath = hf_hub_download(
    repo_id="hf-internal-testing/image-matting-fixtures",
    filename="image.png",
    repo_type="dataset",
)
image = Image.open(filepath).convert("RGB")
filepath = hf_hub_download(
    repo_id="hf-internal-testing/image-matting-fixtures",
    filename="trimap.png",
    repo_type="dataset",
)
trimap = Image.open(filepath).convert("L")

# prepare image + trimap for the model
inputs = processor(images=image, trimaps=trimap, return_tensors="pt")

with torch.no_grad():
    alphas = model(**inputs).alphas
# from IPython import embed; embed()
print(alphas.shape)
