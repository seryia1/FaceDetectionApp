import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

def load_haar_cascade():
    """Load the Haar cascade for face detection"""
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        return face_cascade
    except Exception as e:
        st.error(f"Error loading Haar cascade: {e}")
        return None

def detect_faces(image, face_cascade, scale_factor, min_neighbors):
    """Detect faces in the image using Haar cascade"""
    # Convert PIL image to OpenCV format
    opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=(30, 30)
    )
    
    return faces, opencv_image

def draw_rectangles(image, faces, color):
    """Draw rectangles around detected faces"""
    # Convert hex color to BGR format for OpenCV
    hex_color = color.lstrip('#')
    rgb_color = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    bgr_color = (rgb_color[2], rgb_color[1], rgb_color[0])  # Convert RGB to BGR
    
    # Draw rectangles around faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), bgr_color, 2)
    
    return image

def save_image(image, filename):
    """Save the image with detected faces"""
    try:
        cv2.imwrite(filename, image)
        return True
    except Exception as e:
        st.error(f"Error saving image: {e}")
        return False

def webcam_detection(face_cascade, scale_factor, min_neighbors, rectangle_color):
    """Handle webcam face detection using st.camera_input"""
    st.subheader("📹 Webcam Face Detection")
    
    # Instructions for webcam
    st.markdown("""
    ### 📋 Webcam Instructions:
    1. **Click the camera button below** to take a photo from your webcam
    2. **Allow camera access** when prompted by your browser
    3. **View the results** with detected faces highlighted
    4. **Adjust parameters** in the sidebar and retake if needed
    5. **Save the photo** with detected faces
    
    💡 **Tip:** Make sure you're in good lighting and facing the camera!
    """)
    
    # Use Streamlit's built-in camera input
    img_data = st.camera_input("📷 Click here to capture an image from your webcam")
    
    if img_data is not None:
        # Read the image from BytesIO object
        file_bytes = np.asarray(bytearray(img_data.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces using the same parameters as the main app
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=scale_factor, 
            minNeighbors=min_neighbors,
            minSize=(30, 30)
        )
        
        # Draw rectangles around detected faces
        # Convert hex color to BGR format for OpenCV
        hex_color = rectangle_color.lstrip('#')
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (b, g, r), 2)
        
        # Display the image with detected faces
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 
                caption=f"Webcam photo with {len(faces)} face(s) detected", 
                use_column_width=True)
        
        # Display face count with prominent styling
        if len(faces) > 0:
            st.success(f"🎉 **SUCCESS!** Detected {len(faces)} face(s) in the webcam photo!")
            
            # Show face coordinates
            with st.expander("📍 Face Coordinates Details"):
                for i, (x, y, w, h) in enumerate(faces):
                    st.write(f"**Face {i+1}:** x={x}, y={y}, width={w}, height={h}")
        else:
            st.warning("⚠️ **No faces detected** in the webcam photo. Try adjusting the parameters or retake the photo.")
        
        # Save functionality for webcam capture
        st.markdown("### 💾 Save Webcam Photo")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            webcam_filename = st.text_input(
                "Filename for webcam photo",
                value="webcam_detected_faces.jpg",
                key="webcam_filename"
            )
        
        with col2:
            if st.button("💾 Save Webcam Photo", type="primary"):
                if save_image(image, webcam_filename):
                    st.success(f"✅ Webcam photo saved as '{webcam_filename}'!")
                    
                    # Provide download link
                    with open(webcam_filename, "rb") as file:
                        st.download_button(
                            label="📥 Download Webcam Photo",
                            data=file.read(),
                            file_name=webcam_filename,
                            mime="image/jpeg",
                            key="download_webcam"
                        )
        
        # Statistics for webcam capture
        st.markdown("### 📈 Webcam Detection Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Faces Detected", len(faces))
        
        with col2:
            st.metric("Scale Factor", scale_factor)
        
        with col3:
            st.metric("Min Neighbors", min_neighbors)
        
        with col4:
            frame_height, frame_width = image.shape[:2]
            st.metric("Frame Size", f"{frame_width}x{frame_height}")

def main():
    st.set_page_config(
        page_title="Face Detection App",
        page_icon="👤",
        layout="wide"
    )
    
    st.title("👤 Face Detection App with Haar Cascades")
    st.markdown("---")
    
    # Instructions section
    with st.expander("📋 How to Use This App", expanded=True):
        st.markdown("""
        ### Instructions:
        
        #### 📷 Image Upload Mode:
        1. **Upload an Image**: Use the file uploader to select an image from your device (JPG, JPEG, or PNG)
        2. **View Results**: See the original and processed images side by side with face count
        3. **Save Image**: Download the processed image with detected faces
        
        #### 📹 Webcam Mode:
        1. **Click Camera Button**: Use the built-in camera input to take a photo
        2. **Allow Camera Access**: Grant permission when prompted by your browser
        3. **View Detection**: See faces detected with highlighted rectangles
        4. **Adjust Parameters**: Fine-tune detection settings and retake if needed
        5. **Save Photo**: Download the processed photo with detected faces
        
        #### 🔧 Parameter Adjustment:
        - **Scale Factor**: Controls detection thoroughness (1.1 - 2.0)
        - **Min Neighbors**: Reduces false positives (1 - 10)
        - **Rectangle Color**: Customize face detection rectangle color
        
        ### Tips:
        - **Lower Scale Factor**: More thorough detection but slower processing
        - **Higher Min Neighbors**: Fewer false positives but might miss some faces
        - **Good Lighting**: Ensure adequate lighting for better webcam detection
        - **Camera Permissions**: Allow browser access to your camera for webcam features
        """)
    
    # Load Haar cascade
    face_cascade = load_haar_cascade()
    if face_cascade is None:
        st.error("Failed to load Haar cascade. Please check your OpenCV installation.")
        return
    
    # Sidebar for parameters
    st.sidebar.header("🔧 Detection Parameters")
    
    # Scale factor slider
    scale_factor = st.sidebar.slider(
        "Scale Factor",
        min_value=1.1,
        max_value=2.0,
        value=1.3,
        step=0.1,
        help="How much the image size is reduced at each scale. Lower values = more thorough detection."
    )
    
    # Min neighbors slider
    min_neighbors = st.sidebar.slider(
        "Min Neighbors",
        min_value=1,
        max_value=10,
        value=5,
        step=1,
        help="How many neighbors each face rectangle should retain. Higher values = fewer false positives."
    )
    
    # Color picker
    rectangle_color = st.sidebar.color_picker(
        "Rectangle Color",
        value="#00FF00",
        help="Choose the color for face detection rectangles"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Current Settings")
    st.sidebar.write(f"**Scale Factor:** {scale_factor}")
    st.sidebar.write(f"**Min Neighbors:** {min_neighbors}")
    st.sidebar.write(f"**Rectangle Color:** {rectangle_color}")
    
    # Mode selection
    st.header("🎯 Detection Mode")
    mode = st.radio(
        "Choose detection mode:",
        ["📷 Image Upload", "📹 Webcam Detection"],
        horizontal=True
    )
    
    if mode == "📷 Image Upload":
        st.markdown("---")
        st.header("📷 Image Upload Mode")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png'],
            help="Upload an image to detect faces"
        )
        
        if uploaded_file is not None:
            # Load and display original image
            image = Image.open(uploaded_file)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📷 Original Image")
                st.image(image, caption="Original Image", use_column_width=True)
            
            # Detect faces
            with st.spinner("Detecting faces..."):
                faces, opencv_image = detect_faces(image, face_cascade, scale_factor, min_neighbors)
            
            # Draw rectangles around faces
            processed_image = draw_rectangles(opencv_image.copy(), faces, rectangle_color)
            
            with col2:
                st.subheader("🎯 Detected Faces")
                # Convert back to RGB for display
                processed_image_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
                st.image(processed_image_rgb, caption=f"Detected {len(faces)} face(s)", use_column_width=True)
            
            # Display detection results
            st.markdown("---")
            if len(faces) > 0:
                st.success(f"✅ Successfully detected {len(faces)} face(s)!")
                
                # Show face coordinates
                with st.expander("📍 Face Coordinates"):
                    for i, (x, y, w, h) in enumerate(faces):
                        st.write(f"**Face {i+1}:** x={x}, y={y}, width={w}, height={h}")
            else:
                st.warning("⚠️ No faces detected. Try adjusting the parameters or use a different image.")
            
            # Save functionality
            st.markdown("---")
            st.subheader("💾 Save Processed Image")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                filename = st.text_input(
                    "Filename",
                    value="detected_faces.jpg",
                    help="Enter the filename for saving the processed image"
                )
            
            with col2:
                if st.button("💾 Save Image", type="primary"):
                    if save_image(processed_image, filename):
                        st.success(f"✅ Image saved successfully as '{filename}'!")
                        
                        # Provide download link
                        with open(filename, "rb") as file:
                            btn = st.download_button(
                                label="📥 Download Image",
                                data=file.read(),
                                file_name=filename,
                                mime="image/jpeg"
                            )
            
            # Additional information
            st.markdown("---")
            st.subheader("📈 Detection Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Faces Detected", len(faces))
            
            with col2:
                st.metric("Scale Factor", scale_factor)
            
            with col3:
                st.metric("Min Neighbors", min_neighbors)
            
            with col4:
                st.metric("Image Size", f"{image.size[0]}x{image.size[1]}")
        
        else:
            st.info("👆 Please upload an image to start face detection!")
            
            # Show sample instructions when no image is uploaded
            st.markdown("---")
            st.subheader("🖼️ Sample Usage")
            st.markdown("""
            This app uses **Haar Cascade Classifiers** for face detection, which is a machine learning approach 
            where a cascade function is trained from positive and negative images.
            
            **Key Features:**
            - Real-time parameter adjustment
            - Customizable rectangle colors
            - Image saving functionality
            - Detailed detection statistics
            - User-friendly interface
            """)
    
    elif mode == "📹 Webcam Detection":
        st.markdown("---")
        
        # Webcam detection mode using the working implementation
        webcam_detection(face_cascade, scale_factor, min_neighbors, rectangle_color)

if __name__ == "__main__":
    main()
