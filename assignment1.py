import streamlit as st
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tempfile
import os
import pandas as pd

# Function to calculate PSNR
def psnr(original, denoised):
    mse = np.mean((original - denoised) ** 2)
    if mse == 0:
        return 100
    return 20 * np.log10(255.0 / np.sqrt(mse))

# Function to calculate MSE
def mse(original, denoised):
    return np.mean((original - denoised) ** 2)

# Function to calculate ENL (Equivalent Noise Level)
def enl(image):
    mean = np.mean(image)
    std_dev = np.std(image)
    return mean ** 2 / std_dev ** 2 if std_dev != 0 else 0

# Function to calculate CNR (Contrast-to-Noise Ratio)
def cnr(original, processed):
    signal_mean = np.mean(processed)
    signal_std = np.std(processed)
    background_mean = np.mean(original)
    background_std = np.std(original)
    return abs(signal_mean - background_mean) / np.sqrt(signal_std**2 + background_std**2)

# Function to calculate NM (Noise Metric)
def nm(original, processed):
    return np.sum(np.abs(original - processed)) / original.size

# Function to display image and its histogram
def display_image_and_histogram(image, title):
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(image.T, cmap='gray', origin='lower')
    ax[0].axis("off")
    ax[0].set_title(f"{title} - Image")
    ax[1].hist(image.ravel(), bins=256, range=(0, 255), color='gray', alpha=0.7)
    ax[1].set_title(f"{title} - Histogram")
    ax[1].set_xlim(0, 255)
    st.pyplot(fig)

# Streamlit UI
st.title("5023211056_Muhammad Fakhri Andika Mutiara_Assignment 1")

# File uploader for NIfTI file
nifti_file = st.file_uploader("Upload NIfTI file (.nii)", type=["nii"])

if nifti_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".nii") as temp_file:
        temp_file.write(nifti_file.read())
        temp_file_path = temp_file.name

    try:
        nifti = nib.load(temp_file_path)
        data = nifti.get_fdata()

        pixdim = nifti.header['pixdim']
        rounded_pixdim = np.round(pixdim, 4)

        st.subheader("NIfTI Image Information")
        st.write(f"Shape: {data.shape}")

        nifti_header = nifti.header
        header_info = {
            'dim': nifti_header['dim'],
            'datatype': nifti_header['datatype'],
            'pixdim': rounded_pixdim,
            'sform_code': nifti_header['sform_code'],
            'qform_code': nifti_header['qform_code'],
            'sizeof_hdr': nifti_header['sizeof_hdr'],
            'xdim': nifti_header['dim'][1],
            'ydim': nifti_header['dim'][2],
            'zdim': nifti_header['dim'][3]
        }

        for key, value in header_info.items():
            st.write(f"{key: <15}: {value}")

        data = nifti.get_fdata()

        # Normalize the full 3D data to uint8
        data_min = np.min(data)
        data_max = np.max(data)
        data = ((data - data_min) / (data_max - data_min) * 255).astype(np.uint8)



        st.subheader("Pixel Data Information")
        st.write(f"Pixel Data Type: {data.dtype}")
        st.write(f"Pixel Data Shape: {data.shape}")


                # Display 9 slices as preview
        st.subheader("Preview of 9 Slices from the Volume")
        num_slices = data.shape[2]
        fig, axes = plt.subplots(3, 3, figsize=(10, 10))
        for i, ax in enumerate(axes.flat):
            slice_idx = int(i * num_slices / 9)
            ax.imshow(data[:, :, slice_idx].T, cmap="gray", origin="lower", vmin=0, vmax=255)
            ax.set_title(f"Slice {slice_idx}")
            ax.axis("off")
        st.pyplot(fig)

        # Add slice selection slider (now placed correctly before processing)
        slice_index = st.slider("Select slice to display and process", min_value=0, max_value=num_slices - 1, value=num_slices // 2)

        if st.button("Apply All Transformations and Show Results"):
            imtype = data[:, :, slice_index]



            st.subheader(f"Original Slice {slice_index}")
            fig, ax = plt.subplots(figsize=(6, 6))
            cax = ax.imshow(imtype, cmap='gray', vmin=0, vmax=255)
            ax.axis("off")
            st.pyplot(fig)

            imtype_norm = ((imtype - np.min(imtype)) / (np.max(imtype) - np.min(imtype)) * 255).astype(np.uint8)

            # Histogram Equalization
            hist_eq = cv2.equalizeHist(imtype_norm)
            st.subheader("Histogram Equalization Result")
            display_image_and_histogram(hist_eq, "Histogram Equalization")

            # Adaptive Histogram Equalization (AHE)
            ahe = cv2.createCLAHE(clipLimit=40.0, tileGridSize=(8, 8))  # High clip limit for strong AHE
            ahe_eq = ahe.apply(imtype_norm)
            st.subheader("Adaptive Histogram Equalization Result")
            display_image_and_histogram(ahe_eq, "Adaptive Histogram Equalization")

            # Contrast Limited Adaptive Histogram Equalization (CLAHE)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            clahe_eq = clahe.apply(imtype_norm)
            st.subheader("CLAHE Result")
            display_image_and_histogram(clahe_eq, "CLAHE")

            # Contrast Stretching
            p2, p98 = np.percentile(imtype_norm, (2, 98))
            contrast_stretched = np.clip((imtype_norm - p2) * (255 / (p98 - p2)), 0, 255).astype(np.uint8)
            st.subheader("Contrast Stretching Result")
            display_image_and_histogram(contrast_stretched, "Contrast Stretching")

            # Negative Image
            neg_img = 255 - imtype_norm
            st.subheader("Negative/Inverted Image Result")
            display_image_and_histogram(neg_img, "Negative/Inverted Image")

            performance_metrics = pd.DataFrame(columns=["Technique", "PSNR", "MSE", "ENL", "CNR", "NM"])

            metrics = [
                ("Histogram Equalization", imtype, hist_eq),
                ("Adaptive Histogram Equalization", imtype, ahe_eq),
                ("CLAHE", imtype, clahe_eq),
                ("Contrast Stretching", imtype, contrast_stretched),
                ("Negative/Inverted Image", imtype, neg_img)
            ]

            for technique, original, processed in metrics:
                psnr_val = psnr(original, processed)
                mse_val = mse(original, processed)
                enl_val = enl(processed)
                cnr_val = cnr(original, processed)
                nm_val = nm(original, processed)

                performance_metrics = pd.concat([performance_metrics, pd.DataFrame([{
                    "Technique": technique,
                    "PSNR": psnr_val,
                    "MSE": mse_val,
                    "ENL": enl_val,
                    "CNR": cnr_val,
                    "NM": nm_val
                }])], ignore_index=True)

            st.subheader("Performance Comparison Table")
            st.write(performance_metrics)

    except Exception as e:
        st.error(f"Error loading NIfTI file: {e}")

    os.remove(temp_file_path)
