
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch

from PIL import Image
from transformers import CLIPModel, CLIPImageProcessor


# Application configuration
st.set_page_config(
    page_title="ProtoSight | Website Typicality Analyzer",
    page_icon="🔎",
    layout="centered"
)

MODEL_ID = "openai/clip-vit-base-patch32"
MODEL_DIR = Path("models")


@st.cache_resource(show_spinner="Loading AI models...")
def load_models():
    torch.set_num_threads(2)

    vision_model = CLIPModel.from_pretrained(MODEL_ID)
    image_processor = CLIPImageProcessor.from_pretrained(MODEL_ID)

    vision_model.eval()

    regressors = {
        category: joblib.load(
            MODEL_DIR / f"final_clip_ridge_{category}.joblib"
        )
        for category in ["fashion", "homeware"]
    }

    return vision_model, image_processor, regressors


@st.cache_data
def load_reference_data():
    return pd.read_csv("reference_scores.csv")


def predict_typicality(image, category):
    vision_model, image_processor, regressors = load_models()

    # Match the preprocessing used during CLIP training
    viewport_height = min(
        image.height,
        max(1, int(image.width * 0.75))
    )

    viewport = image.crop(
        (0, 0, image.width, viewport_height)
    )

    inputs = image_processor(
        images=viewport,
        return_tensors="pt"
    )

    with torch.inference_mode():
        embeddings = vision_model.get_image_features(
            pixel_values=inputs["pixel_values"]
        )

        if not isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.pooler_output

    features = embeddings.cpu().numpy()

    score = float(
        regressors[category].predict(features)[0]
    )

    return score


# Main interface
st.title("🔎 ProtoSight")
st.subheader("Can AI predict how familiar a website looks?")

st.write(
    "Upload a website screenshot to estimate its "
    "human-perceived typicality using CLIP and Ridge Regression."
)

st.info(
    "This research prototype currently supports "
    "Fashion and Homeware websites."
)

category = st.selectbox(
    "Website category",
    ["Fashion", "Homeware"]
)

uploaded_file = st.file_uploader(
    "Upload a website screenshot",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.write("### Screenshot preprocessing")

    top_crop = st.slider(
        "Remove browser header (%)",
        min_value=0,
        max_value=25,
        value=0
    )

    bottom_crop = st.slider(
        "Remove taskbar (%)",
        min_value=0,
        max_value=20,
        value=0
    )

    top = int(image.height * top_crop / 100)
    bottom = image.height - int(
        image.height * bottom_crop / 100
    )

    if bottom <= top:
        st.error("Invalid cropping boundaries.")
        st.stop()

    clean_image = image.crop(
        (0, top, image.width, bottom)
    )

    st.image(
        clean_image,
        caption="Website screenshot preview",
        use_container_width=True
    )

    if st.button(
        "Analyze Website",
        type="primary",
        use_container_width=True
    ):

        with st.spinner("Analyzing visual features..."):
            predicted_score = predict_typicality(
                clean_image,
                category.lower()
            )

        reference_data = load_reference_data()

        category_scores = reference_data.loc[
            reference_data["category"] == category.lower(),
            "TYP"
        ].dropna().to_numpy()

        percentile = (
            np.mean(category_scores <= predicted_score) * 100
        )

        st.success("Analysis completed!")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Typicality Score",
                f"{predicted_score:.3f}"
            )

        with col2:
            st.metric(
                "Reference Percentile",
                f"{percentile:.0f} / 100"
            )

        st.write("### Interpretation")

        st.write(
            f"The predicted score is above approximately "
            f"{percentile:.1f}% of historical human typicality "
            f"ratings in the {category.lower()} reference dataset."
        )

        st.caption(
            "The score measures predicted typicality, not "
            "design quality. The percentile is not a "
            "confidence level or accuracy percentage."
        )


st.divider()

st.write("### About the Research")

st.markdown(
    """
    **Research approach:** CLIP ViT-B/32 visual embeddings
    with category-specific Ridge Regression.

    **Dataset:** 1,064 e-commerce website screenshots.

    **Held-out test performance:**
    - MAE: 0.4133
    - R²: 0.2878
    - Spearman correlation: 0.4702
    - 15.05% lower MAE than the mean baseline

    This is an independent experimental research project.
    Results may not generalize to all modern websites.
    """
)

st.caption(
    "Research prototype. Uploaded screenshots are processed "
    "for inference; avoid submitting private or sensitive content."
)
