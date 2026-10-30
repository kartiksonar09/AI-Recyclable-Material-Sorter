
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Recyclable material sorter",
    page_icon=":material/recycling:",
    layout="wide",
)

CLASSES = [
    "cardboard", "glass", "metal",
    "paper", "plastic", "trash"
]

SORTING_GUIDE = {
    "cardboard": "Cardboard recycling stream",
    "glass": "Glass recycling stream",
    "metal": "Metal recycling stream",
    "paper": "Paper recycling stream",
    "plastic": "Plastic recycling stream",
    "trash": "General waste: inspect local rules"
}

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "waste_sorting_model.keras"
    )

st.title("Recyclable material sorter", icon=":material/recycling:")
st.caption(
    "Identify a material from a photo and get a starting point for sorting it."
)

upload_column, result_column = st.columns([0.9, 1.1], gap="large")

with upload_column:
    with st.container(border=True):
        st.subheader("Upload a material", icon=":material/add_photo_alternate:")
        st.caption("Use a clear photo of one item for the best result.")
        uploaded_file = st.file_uploader(
            "Choose an image",
            type=["jpg", "jpeg", "png"],
        )

        identify_clicked = False
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(
                image,
                caption="Image to classify",
                width="stretch",
                alt="Uploaded material to classify",
            )
            identify_clicked = st.button(
                "Identify material",
                type="primary",
                icon=":material/search:",
                width="stretch",
            )

with result_column:
    with st.container(border=True):
        st.subheader("Sorting result", icon=":material/recycling:")

        if uploaded_file is None:
            st.info(
                "Upload a photo to see the predicted material and sorting guidance.",
                icon=":material/info:",
            )
        elif not identify_clicked:
            st.caption("Your result will appear here after analysis.")
        else:
            model = load_model()

            image = image.resize((224, 224))
            arr = np.asarray(image, dtype=np.float32)
            arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
            arr = np.expand_dims(arr, axis=0)

            scores = model.predict(arr, verbose=0)[0]
            index = int(np.argmax(scores))
            material = CLASSES[index]
            confidence = float(scores[index])

            st.success(f"{material.title()} identified", icon=":material/check_circle:")
            st.metric("Model confidence", f"{confidence * 100:.1f}%")
            st.markdown("**Suggested sorting stream**")
            st.write(SORTING_GUIDE[material])

            if confidence < 0.60:
                st.warning(
                    "Low-confidence prediction. Verify the material manually.",
                    icon=":material/warning:",
                )

            with st.expander("View scores for all categories"):
                st.bar_chart(
                    {
                        CLASSES[i]: float(scores[i])
                        for i in range(len(CLASSES))
                    },
                    alt="Model scores for cardboard, glass, metal, paper, plastic, and trash",
                )

st.caption(
    "Visual classification only. Check local recycling rules before disposal."
)
