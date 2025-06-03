import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile

# Configuration de la page
st.set_page_config(page_title="Détection de visages", layout="centered")

st.title("📸 Détection de visages avec Viola-Jones")
st.markdown("""
Bienvenue dans l'application de détection de visages !

**Fonctionnalités :**
- 📁 Téléversement d’image ou capture 🎥 via webcam
- 🎨 Couleur personnalisée des rectangles
- 🔧 Ajustement des paramètres `scaleFactor` et `minNeighbors`
- 💾 Enregistrement de l’image détectée
""")

# Fonction pour convertir une couleur hexadécimale (HTML) en format BGR pour OpenCV
def hex_to_bgr(hex_color):
    hex_color = hex_color.lstrip('#')
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2 ,4))
    return rgb[::-1]  # Conversion RGB ➝ BGR

# Fonction de détection de visages
def detect_faces(image_cv, scaleFactor=1.1, minNeighbors=5, color_bgr=(0, 255, 0)):
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors)
    for (x, y, w, h) in faces:
        cv2.rectangle(image_cv, (x, y), (x+w, y+h), color_bgr, 2)
    return image_cv, len(faces)

# Charger le classifieur de visages de Viola-Jones
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Interface : choix de la source
source = st.radio("📷 Choisissez une source :", ["Téléverser une image", "Utiliser la webcam"])
color = st.color_picker("🎨 Couleur du rectangle", "#00FF00")
scaleFactor = st.slider("🔧 scaleFactor", 1.01, 1.5, 1.1, 0.01)
minNeighbors = st.slider("🔧 minNeighbors", 1, 10, 5)

# 📁 Mode téléversement
if source == "Téléverser une image":
    uploaded_file = st.file_uploader("📁 Téléversez une image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)
        image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        if st.button("✅ Détecter les visages", key="detect_button"):
            result_img, nb_faces = detect_faces(image_cv.copy(), scaleFactor, minNeighbors, hex_to_bgr(color))
            st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), use_column_width=True)
            st.success(f"🔎 {nb_faces} visage(s) détecté(s).")

            if st.checkbox("💾 Enregistrer l'image détectée"):
                cv2.imwrite("visages_detectes.jpg", result_img)
                st.info("✅ Image enregistrée sous 'visages_detectes.jpg'.")

# 🎥 Mode webcam
else:
    st.markdown("⏺ Cliquez sur **Démarrer la capture** pour activer la webcam.")
    run = st.checkbox("Démarrer la webcam")

    FRAME_WINDOW = st.image([])

    if run:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("❌ Webcam non disponible.")
        else:
            while run:
                ret, frame = cap.read()
                if not ret:
                    st.error("❌ Erreur de capture vidéo.")
                    break

                frame_processed, nb_faces = detect_faces(frame.copy(), scaleFactor, minNeighbors, hex_to_bgr(color))
                frame_rgb = cv2.cvtColor(frame_processed, cv2.COLOR_BGR2RGB)
                FRAME_WINDOW.image(frame_rgb)

                if st.button("📸 Capturer & Enregistrer", key="capture_button"):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmpfile:
                        cv2.imwrite(tmpfile.name, frame_processed)
                        st.success(f"✅ Image capturée et enregistrée : {tmpfile.name}")
                    break
            cap.release()
