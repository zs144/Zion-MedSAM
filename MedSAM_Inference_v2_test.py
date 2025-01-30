# -*- coding: utf-8 -*-

"""
usage example:
python MedSAM_Inference.py -i assets/img_demo.png -o ./ --box "[95,255,190,350]"

"""
#%%
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import os

join = os.path.join
import torch
from segment_anything import sam_model_registry
from skimage import io, transform
import torch.nn.functional as F
import argparse
from time import time


def get_bounding_boxes(test_gt_mask, buffer=0, format='xywh'):
    # Convert the image to grayscale
    np_gt_mask = np.array(test_gt_mask)
    # Apply thresholding
    _, thresh = cv2.threshold(np_gt_mask, 127, 255, cv2.THRESH_BINARY)
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Get bounding boxes
    bboxes = [cv2.boundingRect(cnt) for cnt in contours]
    if buffer > 0:
        bboxes = [(x - buffer, y - buffer, w + 2 * buffer, h + 2 * buffer)
                 for x, y, w, h in bboxes]
    if format == 'xyxy':
        bboxes = [(x, y, x+w, y+h) for x, y, w, h in bboxes]
    elif format == 'xywh':
        bboxes = bboxes
    else:
        raise ValueError("The format should be either 'xywh' or 'xyxy'.")
    return bboxes


# visualization functions
# source: https://github.com/facebookresearch/segment-anything/blob/main/notebooks/predictor_example.ipynb
# change color to avoid red and green
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([251 / 255, 252 / 255, 30 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

# NOTE: the original show_box function indicates that the format of bboxes need
# to be (x1, y1, x2, y2), where (x1, y1) is the top left corner and (x2, y2) is
# the bottom right corner.
# The new show_box function below is tailored for multiple input box prompts.
def show_box(bboxes, ax):
    for bbox in bboxes:
        x, y, x2, y2 = bbox
        w = x2 - x
        h = y2 - y
        rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor="green", linewidth=2)
        ax.add_patch(rect)


@torch.no_grad()
def medsam_inference(medsam_model, img_embed, box_1024, H, W):
    box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=img_embed.device)
    if len(box_torch.shape) == 2:
        box_torch = box_torch[:, None, :]  # (B, 1, 4)

    sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
        points=None,
        boxes=box_torch,
        masks=None,
    )
    low_res_logits, _ = medsam_model.mask_decoder(
        image_embeddings=img_embed,  # (B, 256, 64, 64)
        image_pe=medsam_model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
        sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
        dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
        multimask_output=False,
    )

    low_res_pred = torch.sigmoid(low_res_logits)  # (1, 1, 256, 256)

    low_res_pred = F.interpolate(
        low_res_pred,
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    )  # (1, 1, gt.shape)
    low_res_pred = low_res_pred.squeeze().cpu().numpy()  # (256, 256)
    medsam_seg = (low_res_pred > 0.5).astype(np.uint8)
    # fig, ax = plt.subplots(nrows=1, ncols=8, figsize=(5, 40))
    # for i in range(medsam_seg.shape[0]):
    #     ax[i].imshow(medsam_seg[i,:,:])

    if len(medsam_seg.shape) > 2:
        agg_medsam_seg = np.any(medsam_seg == 1, axis=0).astype(np.uint8)
    else:
        agg_medsam_seg = medsam_seg
    return agg_medsam_seg
#%%
start = time()
image_path  = "/hpc/group/yizhanglab/zs144/Zion-ZhangLab/experiments/EXP003/images/patch_6_8.png"
seg_path    = "/hpc/group/yizhanglab/zs144/Zion-ZhangLab/experiments/EXP003/images/"
gt_mask_path= "/hpc/group/yizhanglab/zs144/Zion-ZhangLab/experiments/EXP003/images/gt_mask_6_8.png"
device      = "cuda:0"
checkpoint  = "/hpc/group/yizhanglab/zs144/resources/MedSAM/original_ckpt/medsam_vit_b.pth"

medsam_model = sam_model_registry["vit_b"](checkpoint=checkpoint)
medsam_model = medsam_model.to(device)
medsam_model.eval()

img_np = io.imread(image_path)
if len(img_np.shape) == 2:
    img_3c = np.repeat(img_np[:, :, None], 3, axis=-1)
else:
    img_3c = img_np
H, W, _ = img_3c.shape
gt_mask = io.imread(gt_mask_path)

img_1024 = transform.resize(
    img_3c, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True
).astype(np.uint8)
img_1024 = (img_1024 - img_1024.min()) / np.clip(
    img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
)  # normalize to [0, 1], (H, W, 3)
# convert the shape to (3, H, W)
img_1024_tensor = (
    torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(device)
)

bboxes_np = get_bounding_boxes(gt_mask, buffer=40, format='xyxy')
# transfer box_np t0 1024x1024 scale
bboxes_1024 = bboxes_np / np.array([W, H, W, H]) * 1024
with torch.no_grad():
    image_embedding = medsam_model.image_encoder(img_1024_tensor)  # (1, 256, 64, 64)

medsam_seg = medsam_inference(medsam_model, image_embedding, bboxes_1024, H, W)
medsam_seg = (medsam_seg * 255).astype(np.uint8)
io.imsave(
    join(seg_path, "seg_" + os.path.basename(image_path)),
    medsam_seg,
    check_contrast=False,
)

end = time()
duration = (end - start) / 60
print(f"Time elapse: {duration:.2f} min.")
#%%
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(img_3c)
show_box(bboxes_np, ax[0])
ax[0].set_title("Input Image and Bounding Box")
ax[1].imshow(img_3c)
show_mask(medsam_seg, ax[1])
show_box(bboxes_np, ax[1])
ax[1].set_title("MedSAM Segmentation")
plt.show()
# %%
