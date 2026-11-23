import os
import streamlit as st
import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN
from PIL import Image
from torchvision import transforms

from model import FaceClassifier

DEFAULT_WEIGHTS = os.path.join("weights", "Face_Detector_v1_Gold.pth")
FALLBACK_WEIGHTS = "best_model.pth"


@st.cache_resource
def load_resources():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights_path = DEFAULT_WEIGHTS if os.path.exists(DEFAULT_WEIGHTS) else FALLBACK_WEIGHTS
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Model weights not found at '{DEFAULT_WEIGHTS}' or '{FALLBACK_WEIGHTS}'."
        )

    classifier = FaceClassifier(num_classes=2).to(device)
    classifier.load_state_dict(torch.load(weights_path, map_location=device))
    classifier.eval()

    mtcnn_detector = MTCNN(keep_all=False, device="cpu")
    return classifier, mtcnn_detector, device, weights_path


val_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

st.title("Real vs. AI-Generated Face Detector")
st.caption("MTCNN face extraction + ResNet-18 binary classifier (Discontinued Baseline)")

model, mtcnn, device, loaded_weights = load_resources()
st.sidebar.info(f"Loaded weights: `{loaded_weights}`\nDevice: `{device}`")

uploaded_file = st.file_uploader("Upload a face image...", type=["jpg", "png", "jpeg", "webp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    boxes, _ = mtcnn.detect(image)

    if boxes is None:
        st.error("No human face detected in this image. Please upload a clear portrait.")
    else:
        st.success("Face detected successfully!")

        box = boxes[0]
        left = max(0, int(box[0]))
        top = max(0, int(box[1]))
        right = min(image.width, int(box[2]))
        bottom = min(image.height, int(box[3]))
        cropped_image = image.crop((left, top, right, bottom))

        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Image", use_container_width=True)
        with col2:
            st.image(cropped_image, caption="MTCNN Cropped Face", use_container_width=True)

        input_tensor = val_transform(cropped_image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            probs = F.softmax(output, dim=1)
            confidence, pred = torch.max(probs, 1)

        # Class mapping from ImageFolder: 0 = fake_faces, 1 = real_faces
        conf_pct = confidence.item() * 100
        if pred.item() == 0:
            st.error(f"Prediction: FAKE (AI-Generated) - Confidence: {conf_pct:.2f}%")
        else:
            st.success(f"Prediction: REAL (Human) - Confidence: {conf_pct:.2f}%")
