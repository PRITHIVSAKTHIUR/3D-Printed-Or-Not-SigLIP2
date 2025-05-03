import gradio as gr
from transformers import AutoImageProcessor, SiglipForImageClassification
from PIL import Image
import torch

# Load model and processor
model_name = "prithivMLmods/3D-Printed-Or-Not-SigLIP2"  # Replace with your model path if different
model = SiglipForImageClassification.from_pretrained(model_name)
processor = AutoImageProcessor.from_pretrained(model_name)

# Label mapping
id2label = {
    "0": "3D Printed",
    "1": "Not 3D Printed"
}

def classify_3d_printed(image):
    image = Image.fromarray(image).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()

    prediction = {
        id2label[str(i)]: round(probs[i], 3) for i in range(len(probs))
    }

    return prediction

# Gradio Interface
iface = gr.Interface(
    fn=classify_3d_printed,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Label(num_top_classes=2, label="3D Printing Classification"),
    title="3D-Printed-Or-Not-SigLIP2",
    description="Upload an image to detect if the object is 3D printed or not."
)

if __name__ == "__main__":
    iface.launch()
