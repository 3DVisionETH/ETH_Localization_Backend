import torch
import torchvision.transforms as T
from skimage import io
from skimage.transform import resize
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from PIL import Image

from model import *

# Load the image
image_dir = os.path.join("data", "images")
image_path = image_dir + "/CAB_E_VF_20230609-2-1.png"
image = io.imread(image_path)
name2idx = {'pad': -1, 'doors': 0, 'stairs': 1}
idx2name = {v:k for k, v in name2idx.items()}

# Preprocess the image
img_height, img_width = 500, 700 # Ensure these match the model's expected input size
image_resized = resize(image, (img_height, img_width))
image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).unsqueeze(0).float()

# Move the image to the appropriate device
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
image_tensor = image_tensor.to(device)

# Load the model (assuming the model has been saved and TwoStageDetector class is defined as per your code)
model_save_path = "model.pt"

model = torchvision.models.resnet50(pretrained=True)

req_layers = list(model.children())[:8]
backbone = nn.Sequential(*req_layers).to(device)

# Ensure the parameters are set to require gradients
for param in backbone.named_parameters():
    param[1].requires_grad = True

# Pass the image tensor through the backbone to get the output dimensions
with torch.no_grad():
    out = backbone(image_tensor)

# Extract the dimensions of the output feature map
out_c, out_h, out_w = out.size(dim=1), out.size(dim=2), out.size(dim=3)

detector = TwoStageDetector(img_size=(img_height, img_width), out_size=(out_h, out_w), out_channels=out_c, n_classes=len(name2idx) - 1, roi_size=(2, 2))
detector.load_state_dict(torch.load(model_save_path))
detector.to(device)
detector.eval()  # Set the model to evaluation mode

# Run inference
proposals_final, conf_scores_final, classes_final = detector.inference(image_tensor, conf_thresh=0.9, nms_thresh=0.3)

#print(proposals_final, conf_scores_final, classes_final)

# Visualize the results
#fig, ax = plt.subplots(1, figsize=(12, 8))
#ax.imshow(image)
width_scale_factor = img_width // out_w
height_scale_factor = img_height // out_h
prop_proj_1 = project_bboxes(proposals_final[0], width_scale_factor, height_scale_factor, mode='a2p')
classes_pred_1 = [idx2name[cls] for cls in classes_final[0].tolist()]

print (prop_proj_1,classes_pred_1 )

fig, ax = plt.subplots(1, figsize=(12, 8))
#image = Image.open(image_path).convert("RGB")
ax.imshow(image_resized)

#fig, _ = display_bbox2(prop_proj_1, fig, axes[0], color='green', classes=classes_pred_2)

fig, _ = display_bbox_doors(prop_proj_1, fig, ax, classes=classes_pred_1)

"""
for bbox, label in zip(prop_proj_1, classes_pred_1):
    x_min, y_min, x_max, y_max = bbox
    width = x_max - x_min
    height = y_max - y_min

    # Create a Rectangle patch
    rect = patches.Rectangle((x_min, y_min), width, height, linewidth=1, edgecolor='r', facecolor='none')

    # Add the patch to the Axes
    ax.add_patch(rect)
    ax.text(x_min, y_min, label, color='white', fontsize=12, bbox=dict(facecolor='red', alpha=0.5))

"""


plt.savefig("prediction_doors.png")
plt.show()
