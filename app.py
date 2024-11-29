# Set page configuration first
import streamlit as st
st.set_page_config(page_title="Diabetic Retinopathy Detection", page_icon=":eye:", layout="wide")

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
import cv2
from PIL import Image
import io
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from datetime import date
import os

# Load the trained models
@st.cache_resource
def load_models():
    binary_model = load_model('B-DR-MobileNet2V.h5')
    multi_model = load_model('M-DR-MobileNet2V.h5')
    return binary_model, multi_model

# Load models
binary_model, multi_model = load_models()

# Define class names
binary_class_names = np.array(['DR', 'No_DR'])
multi_class_names = np.array(['Mild', 'Moderate', 'Proliferate_Dr', 'Severe'])

# Helper function to process the image
def process_image(file):
    image = Image.open(file)
    image = image.resize((224, 224))
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    return image

# Function to generate PDF report
def generate_pdf_report(patient_info, binary_prediction, multi_prediction, uploaded_image):
    # Create a unique filename for the PDF
    filename = f"DR_Report_{patient_info['name']}_{date.today()}.pdf"
    
    # Create PDF
    doc = SimpleDocTemplate(filename, pagesize=letter)
    
    # Styles
    styles = getSampleStyleSheet()
    title_style = styles['Title']
    heading_style = styles['Heading2']
    normal_style = styles['Normal']
    
    # Content elements
    content = []
    
    # Title
    content.append(Paragraph("Diabetic Retinopathy Detection Report", title_style))
    content.append(Spacer(1, 12))
    
    # Patient Information Table
    patient_data = [
        ['Patient Detail', 'Information'],
        ['Name', patient_info['name']],
        ['Age', str(patient_info['age'])],
        ['Gender', patient_info['gender']],
        ['Date of Birth', str(patient_info['dob'])],
        ['Family Diabetic Background', patient_info['family_diabetic']]
    ]
    
    patient_table = Table(patient_data, colWidths=[2*inch, 4*inch])
    patient_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.grey),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 12),
        ('BOTTOMPADDING', (0,0), (-1,0), 12),
        ('BACKGROUND', (0,1), (-1,-1), colors.beige),
        ('GRID', (0,0), (-1,-1), 1, colors.black)
    ]))
    content.append(patient_table)
    content.append(Spacer(1, 12))
    
    # Prediction Results
    content.append(Paragraph("Prediction Results", heading_style))
    
    # Prediction Table
    prediction_data = [
        ['Prediction Type', 'Result'],
        ['Binary Classification', binary_prediction],
    ]
    
    if multi_prediction:
        prediction_data.append(['Multi-class Classification', multi_prediction])
    
    prediction_table = Table(prediction_data, colWidths=[2*inch, 4*inch])
    prediction_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.grey),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 12),
        ('BOTTOMPADDING', (0,0), (-1,0), 12),
        ('BACKGROUND', (0,1), (-1,-1), colors.lightblue),
        ('GRID', (0,0), (-1,-1), 1, colors.black)
    ]))
    content.append(prediction_table)
    content.append(Spacer(1, 12))
    
    # Add uploaded image
    if uploaded_image:
        content.append(Paragraph("Uploaded Retinal Image", heading_style))
        # Convert PIL Image to ReportLab Image
        img_byte_arr = io.BytesIO()
        uploaded_image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        
        # Scale image to fit the page
        image = RLImage(io.BytesIO(img_byte_arr), width=4*inch, height=4*inch)
        content.append(image)
    
    # Build PDF
    doc.build(content)
    
    return filename

# Custom CSS for styling
def local_css():
    st.markdown("""
    <style>
    .main-title {
        color: #2C3E50;
        text-align: center;
        font-size: 2.5em;
        margin-bottom: 20px;
    }
    .sidebar .sidebar-content {
        background-color: #F0F4F8;
    }
    .stButton>button {
        background-color: #3498DB;
        color: white;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #2980B9;
        transform: scale(1.05);
    }
    </style>
    """, unsafe_allow_html=True)

# Streamlit app main layout
def main():
    # Apply custom CSS
    local_css()
    
    st.markdown('<h1 class="main-title">Diabetic Retinopathy Detection</h1>', unsafe_allow_html=True)
    
    # Sidebar for patient information
    st.sidebar.header("Patient Information")
    
    # Patient Information Form
    with st.sidebar.form(key='patient_form'):
        name = st.text_input("Patient Name", help="Enter full name of the patient")
        age = st.number_input("Age", min_value=0, max_value=120, help="Patient's age")
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        dob = st.date_input("Date of Birth", help="Patient's date of birth")
        family_diabetic = st.radio("Family Diabetic Background", ["Yes", "No"])
        
        submit_button = st.form_submit_button("Save Patient Details")
        
        # Store patient info in session state
        if submit_button:
            st.session_state.patient_info = {
                'name': name,
                'age': age,
                'gender': gender,
                'dob': dob,
                'family_diabetic': family_diabetic
            }
            st.sidebar.success("Patient details saved successfully!")
    
    # Main content area for image upload and prediction
    st.header("Upload Retinal Image")
    
    # Check if patient details are saved
    if 'patient_info' not in st.session_state:
        st.warning("Please fill out patient details in the sidebar first.")
        return
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Process and display image
        image = process_image(uploaded_file)
        st.image(uploaded_file, caption='Uploaded Retinal Image', use_column_width=True)
        
        # Predictions
        st.subheader('Prediction Results')
        
        # Binary Classification
        binary_prediction = binary_model.predict(image)
        binary_class = binary_class_names[np.argmax(binary_prediction)]
        st.write(f'**Binary Classification:** {binary_class}')
        
        # Multi-class Classification (if DR is detected)
        multi_prediction = None
        if binary_class == 'DR':
            multi_pred = multi_model.predict(image)
            multi_prediction = multi_class_names[np.argmax(multi_pred)]
            st.write(f'**Multi-class Classification:** {multi_prediction}')
        
        # PDF Download Button
        if st.button('Download Prediction Report'):
            try:
                pdf_filename = generate_pdf_report(
                    st.session_state.patient_info, 
                    binary_class, 
                    multi_prediction,
                    Image.open(uploaded_file)  # Pass the uploaded image
                )
                with open(pdf_filename, "rb") as pdf_file:
                    st.download_button(
                        label="Click to Download PDF Report",
                        data=pdf_file,
                        file_name=pdf_filename,
                        mime='application/pdf'
                    )
                os.remove(pdf_filename)  # Clean up temporary file
            except Exception as e:
                st.error(f"Error generating PDF: {e}")

# Run the app
if __name__ == "__main__":
    main()