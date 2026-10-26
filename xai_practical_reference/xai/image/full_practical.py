"""Complete image XAI practical from the uploaded notebook."""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models
from captum.attr import IntegratedGradients, LayerGradCam, LayerAttribution, GuidedBackprop, GuidedGradCam, visualization as viz
from lime.lime_image import LimeImageExplainer
import saliency.core as saliency

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights).to(device); model.eval()
preprocess = weights.transforms()
# Change this to the path of an image on your computer.
image = Image.open("/content/cat.jpg").convert("RGB").resize((224, 224))
image_np = np.asarray(image).astype(np.float32) / 255.0
input_tensor = preprocess(image).unsqueeze(0).to(device); input_tensor.requires_grad_(True)
with torch.no_grad():
    output = model(input_tensor); predicted_class = output.argmax(dim=1).item()
print("Predicted class:", weights.meta["categories"][predicted_class])

def plot_attr(attr, title, sign="absolute_value", method="blended_heat_map", original=image_np):
    if isinstance(attr, torch.Tensor): attr = attr.detach().cpu().squeeze().numpy()
    if attr.ndim == 3 and attr.shape[0] in (1, 3): attr = np.transpose(attr, (1, 2, 0))
    if attr.ndim == 3 and attr.shape[-1] == 1: attr = attr.squeeze(-1)
    viz.visualize_image_attr(attr, original_image=original, method=method, sign=sign, title=title, show_colorbar=False, use_pyplot=True)

# 1. Integrated Gradients
ig = IntegratedGradients(model)
ig_attr = ig.attribute(input_tensor, baselines=torch.zeros_like(input_tensor), target=predicted_class, n_steps=30)
plot_attr(ig_attr, "Integrated Gradients"); print("Integrated Gradients completed")

# 2. XRAI
def call_model_function(images, call_model_args=None, expected_keys=None):
    x = torch.tensor(images, dtype=torch.float32, device=device).permute(0, 3, 1, 2).requires_grad_(True)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    logits = model((x - mean) / std); target = call_model_args["target"]
    gradients = torch.autograd.grad(logits[:, target].sum(), x)[0]
    return {saliency.base.INPUT_OUTPUT_GRADIENTS: gradients.permute(0, 2, 3, 1).detach().cpu().numpy()}
xrai = saliency.XRAI(); xrai_attr = xrai.GetMask(image_np, call_model_function, {"target": predicted_class})
plot_attr(xrai_attr, "XRAI - Important Regions", sign="positive"); print("XRAI completed")

# 3. Grad-CAM
target_layer = model.layer4[-1]; gradcam = LayerGradCam(model, target_layer)
cam_attr = gradcam.attribute(input_tensor, target=predicted_class, relu_attributions=True)
cam_attr = LayerAttribution.interpolate(cam_attr, input_tensor.shape[2:], interpolate_mode="bilinear")
plot_attr(cam_attr, "Grad-CAM", sign="positive"); print("Grad-CAM completed")

# 4. LIME
def predict_for_lime(images):
    batch = torch.stack([preprocess(Image.fromarray(img.astype(np.uint8))) for img in images]).to(device)
    with torch.no_grad(): return F.softmax(model(batch), dim=1).cpu().numpy()
lime_explainer = LimeImageExplainer()
lime_exp = lime_explainer.explain_instance((image_np * 255).astype(np.uint8), predict_for_lime, top_labels=1, num_samples=500)
lime_image, lime_mask = lime_exp.get_image_and_mask(predicted_class, positive_only=True, num_features=5, hide_rest=False)
plot_attr(lime_mask.astype(float), "LIME - Important Superpixels", sign="positive", method="heat_map", original=lime_image.astype(np.float32) / 255.0)
print("LIME completed")

# 5. Guided Backpropagation
guided_bp = GuidedBackprop(model); gb_attr = guided_bp.attribute(input_tensor, target=predicted_class)
plot_attr(gb_attr, "Guided Backpropagation", sign="absolute_value", method="heat_map", original=None); print("Guided Backpropagation completed")

# 6. Guided Grad-CAM
guided_gc = GuidedGradCam(model, target_layer); ggc_attr = guided_gc.attribute(input_tensor, target=predicted_class)
plot_attr(ggc_attr, "Guided Grad-CAM", sign="absolute_value", method="heat_map", original=None)
print("Guided Grad-CAM completed\nAll six explanations generated!")
