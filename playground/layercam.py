import torch
from torchcam.methods import LayerCAM
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt

# 1. Instantiate your custom model
# Assuming your class is named MicroscopyModel
model = MicroscopyModel(num_classes=4).eval()

# 2. Initialize LayerCAM with the explicit target layer path
# We point it specifically to layer2 inside your feature_extractor graph
cam_extractor = LayerCAM(model, target_layer="feature_extractor.layer2")

# 3. Create a dummy image (Replace with a real false-positive image from your dataset)
input_tensor = torch.rand(1, 3, 224, 224)

# 4. Forward pass
out = model(input_tensor)

# 5. Generate CAM for the predicted class
# (Or manually set predicted_class to the class causing the false positives, e.g., 1)
predicted_class = out.squeeze(0).argmax().item()
cams = cam_extractor(predicted_class, out)

# 6. Visualize
original_image = to_pil_image(input_tensor.squeeze(0))
result = overlay_mask(original_image, to_pil_image(cams[0], mode="F"), alpha=0.5)

plt.imshow(result)
plt.axis("off")
plt.title(f"LayerCAM for Class {predicted_class}")
plt.show()

# Clean up hooks
cam_extractor.remove_hooks()
