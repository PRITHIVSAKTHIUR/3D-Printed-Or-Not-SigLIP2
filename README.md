![2.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/Z9102o2b66hGOm_ESlyOP.png)

# **3D-Printed-Or-Not-SigLIP2**

> **3D-Printed-Or-Not-SigLIP2** is a vision-language encoder model fine-tuned from **google/siglip2-base-patch16-224** for **binary image classification**. It is trained to distinguish between images of **3D printed** and **non-3D printed** objects using the **SiglipForImageClassification** architecture.


> [!note]
*SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features* https://arxiv.org/pdf/2502.14786

```py
Classification Report:
                precision    recall  f1-score   support

    3D Printed     0.9108    0.9388    0.9246     25760
Not 3D Printed     0.9368    0.9081    0.9222     25760

      accuracy                         0.9234     51520
     macro avg     0.9238    0.9234    0.9234     51520
  weighted avg     0.9238    0.9234    0.9234     51520
  ```

![download.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/G1DWP3rDDJ_pO4SBlPbXR.png)


---

## **Label Space: 2 Classes**

The model classifies each image into one of the following categories:

```
Class 0: "3D Printed"
Class 1: "Not 3D Printed"
```

---

## **Install Dependencies**

```bash
pip install -q transformers torch pillow gradio
```

---

## **Inference Code**

```python
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
```

---

## **Intended Use**

**3D-Printed-Or-Not-SigLIP2** can be used for:

- **Manufacturing Verification** – Classify objects to ensure they meet production standards.
- **Educational Tools** – Train models and learners to distinguish between manufacturing methods.
- **Retail Filtering** – Categorize product images by manufacturing technique.
- **Quality Control** – Spot check datasets or content for 3D printing.
