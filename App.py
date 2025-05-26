import cv2
import streamlit as st
import uuid
import os

# Load the pre-trained Haar Cascade classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')



def detect_faces(scaleFactor, minNeighbors, color_bgr):
    cap = cv2.VideoCapture(0)
    st.info("Appuyez sur 'q' pour quitter la détection ou 's' pour sauvegarder une image.")

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Échec de la lecture de la webcam.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), color_bgr, 2)

        cv2.imshow('Détection de visages - Appuyez sur q pour quitter', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"face_snapshot_{uuid.uuid4().hex[:6]}.png"
            cv2.imwrite(filename, frame)
            st.success(f"Image sauvegardée : {filename}")

    cap.release()
    cv2.destroyAllWindows()


def app():
    st.title("Détection de visages avec l'algorithme de Viola-Jones")

    st.markdown("""
    ### Instructions:
    - Cliquez sur le bouton **Detect Faces** pour activer la webcam.
    - Une fenêtre s'ouvrira affichant la vidéo en temps réel.
    - Appuyez sur **'q'** pour quitter.
    - Appuyez sur **'s'** pour sauvegarder une image avec visages détectés.
    """)

    color_hex = st.color_picker("Choisissez la couleur des rectangles", "#00FF00")
    color_bgr = tuple(int(color_hex.lstrip('#')[i:i+2], 16) for i in (4, 2, 0))  # HEX to BGR

    scaleFactor = st.slider("Facteur d'échelle (scaleFactor)", 1.05, 2.0, 1.3, 0.05)
    minNeighbors = st.slider("Min Neighbors", 1, 10, 5, 1)

    if st.button("Detect Faces"):
        detect_faces(scaleFactor, minNeighbors, color_bgr)


if __name__ == "__main__":
    app()
