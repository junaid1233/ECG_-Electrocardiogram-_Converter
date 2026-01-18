import streamlit as st
import cv2
import numpy as np
import pandas as pd
from scipy import interpolate
import matplotlib.pyplot as plt
from PIL import Image
import io
import os
import tempfile

# Set page configuration
st.set_page_config(
    page_title="ECG Image to Signal Converter",
    page_icon="❤️",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #e63946;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #1d3557;
        margin-top: 2rem;
    }
    .info-box {
        background-color: #f1faee;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #457b9d;
        margin: 1rem 0;
    }
    .stButton > button {
        background-color: #e63946;
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 5px;
    }
    .stButton > button:hover {
        background-color: #d62828;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header">ECG Image to Numerical Signal Converter</h1>', unsafe_allow_html=True)
st.markdown("""
<div class="info-box">
This tool converts ECG waveform images from the PhysioNet dataset into numerical signals. 
Upload an ECG image with a red waveform and grid, and the tool will extract the signal data and save it as an Excel file.
</div>
""", unsafe_allow_html=True)

# Create columns for layout
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown('<h3 class="sub-header">Upload ECG Image</h3>', unsafe_allow_html=True)

    # File uploader
    uploaded_file = st.file_uploader("Choose an ECG image file",
                                     type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'])

    # Parameters section
    st.markdown('<h3 class="sub-header">Processing Parameters</h3>', unsafe_allow_html=True)

    # Color detection parameters
    st.markdown("**Color Detection**")
    col_a, col_b = st.columns(2)
    with col_a:
        red_lower = st.slider("Red Lower Threshold", 0, 255, 100, key="red_lower")
    with col_b:
        red_upper = st.slider("Red Upper Threshold", 0, 255, 255, key="red_upper")

    # Grid detection parameters
    st.markdown("**Grid Detection**")
    grid_sensitivity = st.slider("Grid Sensitivity", 0.1, 1.0, 0.5,
                                 help="Lower values detect more grid lines, higher values detect only strong lines")

    # Morphology parameters
    st.markdown("**Morphology Operations**")
    morph_kernel = st.slider("Morphology Kernel Size", 1, 10, 3,
                             help="Size of kernel for cleaning operations")

    # Display assumptions
    with st.expander("📋 Assumptions"):
        st.markdown("""
        1. ECG waveform is drawn in **red color**
        2. Grid consists of **small red boxes**
        3. X-axis: 1 small grid box = **1 millisecond**
        4. Y-axis: Grid boxes represent **equal amplitude scale**
        5. Time starts from origin (0,0)
        6. First entry: time=0 ms, amplitude=0
        7. Output will have uniform time intervals of 1 ms
        """)


def detect_red_color(image, lower_thresh=100, upper_thresh=255):
    """
    Detect red color in the image using HSV color space
    """
    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define range for red color (two ranges for hue wrap-around)
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])

    # Create masks for red color
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    # Apply additional threshold based on intensity
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    intensity_mask = cv2.inRange(gray, lower_thresh, upper_thresh)

    # Combine masks
    combined_mask = cv2.bitwise_and(red_mask, intensity_mask)

    return combined_mask


def detect_grid_lines(binary_image, sensitivity=0.5):
    """
    Detect grid lines using Hough Transform
    """
    # Apply edge detection
    edges = cv2.Canny(binary_image, 50, 150, apertureSize=3)

    # Detect lines using Hough Transform
    lines = cv2.HoughLines(edges, 1, np.pi / 180,
                           int(sensitivity * min(binary_image.shape[:2])))

    horizontal_lines = []
    vertical_lines = []

    if lines is not None:
        for line in lines:
            rho, theta = line[0]

            # Classify as horizontal (theta close to 0 or pi) or vertical (theta close to pi/2)
            if abs(theta) < 0.1 or abs(theta - np.pi) < 0.1:
                horizontal_lines.append(rho)
            elif abs(theta - np.pi / 2) < 0.1:
                vertical_lines.append(rho)

    return horizontal_lines, vertical_lines


def extract_waveform(mask, morph_kernel_size=3):
    """
    Extract the ECG waveform from the mask
    """
    # Clean up the mask using morphological operations
    kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

    # Find contours
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, None

    # Find the largest contour (assuming it's the waveform)
    largest_contour = max(contours, key=cv2.contourArea)

    # Simplify contour (reduce points)
    epsilon = 0.001 * cv2.arcLength(largest_contour, True)
    simplified = cv2.approxPolyDP(largest_contour, epsilon, True)

    # Extract x and y coordinates
    points = simplified.reshape(-1, 2)
    x_coords = points[:, 0]
    y_coords = points[:, 1]

    return x_coords, y_coords


def calculate_grid_spacing(horizontal_lines, vertical_lines, image_shape):
    """
    Calculate grid spacing from detected lines
    """
    # Sort lines and calculate spacing
    horizontal_spacing = None
    vertical_spacing = None

    if len(horizontal_lines) >= 2:
        sorted_horizontal = sorted(horizontal_lines)
        spacings = np.diff(sorted_horizontal)
        horizontal_spacing = np.median(spacings[spacings > 0])

    if len(vertical_lines) >= 2:
        sorted_vertical = sorted(vertical_lines)
        spacings = np.diff(sorted_vertical)
        vertical_spacing = np.median(spacings[spacings > 0])

    return horizontal_spacing, vertical_spacing


def scale_coordinates(x_pixels, y_pixels, grid_spacing_x, grid_spacing_y, image_height):
    """
    Convert pixel coordinates to time (ms) and amplitude values
    """
    if grid_spacing_x is None or grid_spacing_y is None:
        # Default scaling if grid detection fails
        grid_spacing_x = 10  # pixels per ms (assumed)
        grid_spacing_y = 10  # pixels per amplitude unit (assumed)
        st.warning("Grid detection incomplete. Using default scaling.")

    # Convert to time (1 grid cell = 1 ms)
    time_ms = x_pixels / grid_spacing_x

    # Convert to amplitude (invert y-axis as image coordinates start from top)
    amplitude = (image_height - y_pixels) / grid_spacing_y

    return time_ms, amplitude


def interpolate_signal(time_ms, amplitude, max_time=None):
    """
    Interpolate the signal to get uniform time intervals of 1 ms
    """
    # Remove any duplicate time points
    unique_indices = np.unique(time_ms, return_index=True)[1]
    time_unique = time_ms[unique_indices]
    amp_unique = amplitude[unique_indices]

    # Sort by time
    sort_indices = np.argsort(time_unique)
    time_sorted = time_unique[sort_indices]
    amp_sorted = amp_unique[sort_indices]

    # Determine interpolation range
    min_time = 0
    max_time_interp = int(np.ceil(time_sorted.max())) if max_time is None else max_time

    # Create interpolation function
    interpolation_func = interpolate.interp1d(
        time_sorted,
        amp_sorted,
        kind='cubic',
        fill_value='extrapolate',
        bounds_error=False
    )

    # Generate uniform time grid
    uniform_time = np.arange(min_time, max_time_interp + 1)

    # Interpolate amplitude values
    uniform_amplitude = interpolation_func(uniform_time)

    # Ensure first point is (0, 0)
    uniform_amplitude[0] = 0

    return uniform_time, uniform_amplitude


def process_ecg_image(image_array, red_lower=100, red_upper=255,
                      grid_sensitivity=0.5, morph_kernel=3):
    """
    Main processing function for ECG image
    """
    # Step 1: Detect red color regions
    mask = detect_red_color(image_array, red_lower, red_upper)

    # Step 2: Detect grid lines
    horizontal_lines, vertical_lines = detect_grid_lines(mask, grid_sensitivity)

    # Step 3: Extract waveform
    x_pixels, y_pixels = extract_waveform(mask, morph_kernel)

    if x_pixels is None or y_pixels is None:
        raise ValueError("Could not detect ECG waveform in the image")

    # Step 4: Calculate grid spacing
    grid_spacing_x, grid_spacing_y = calculate_grid_spacing(
        horizontal_lines, vertical_lines, image_array.shape[:2]
    )

    # Step 5: Scale coordinates
    time_ms, amplitude = scale_coordinates(
        x_pixels, y_pixels,
        grid_spacing_x, grid_spacing_y,
        image_array.shape[0]
    )

    # Step 6: Interpolate to uniform time intervals
    uniform_time, uniform_amplitude = interpolate_signal(time_ms, amplitude)

    # Step 7: Create DataFrame
    df = pd.DataFrame({
        'Time_ms': uniform_time,
        'Amplitude': uniform_amplitude
    })

    return df, mask, (x_pixels, y_pixels), (horizontal_lines, vertical_lines)


# Main processing
if uploaded_file is not None:
    # Display original image
    with col1:
        st.image(uploaded_file, caption="Uploaded ECG Image", use_column_width=True)

    # Process button
    process_button = st.button("🚀 Process ECG Image", use_container_width=True)

    if process_button:
        try:
            # Read and convert image
            image_bytes = uploaded_file.read()
            image_array = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Show processing status
            with st.spinner('Processing ECG image...'):
                # Process the image
                df, mask, waveform_points, grid_lines = process_ecg_image(
                    image, red_lower, red_upper, grid_sensitivity, morph_kernel
                )

            # Display results in second column
            with col2:
                st.markdown('<h3 class="sub-header">Processing Results</h3>', unsafe_allow_html=True)

                # Display statistics
                col_stats1, col_stats2, col_stats3 = st.columns(3)
                with col_stats1:
                    st.metric("Total Samples", len(df))
                with col_stats2:
                    st.metric("Duration (ms)", f"{df['Time_ms'].max():.0f}")
                with col_stats3:
                    st.metric("Amplitude Range", f"{df['Amplitude'].min():.2f} to {df['Amplitude'].max():.2f}")

                # Preview of extracted data
                st.markdown("**Preview of Extracted Signal:**")
                st.dataframe(df.head(10), use_container_width=True)

                # Plot extracted signal
                st.markdown("**Visualization:**")
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))

                # Original image
                axes[0, 0].imshow(image_rgb)
                axes[0, 0].set_title('Original ECG Image')
                axes[0, 0].axis('off')

                # Detected red regions
                axes[0, 1].imshow(mask, cmap='gray')
                axes[0, 1].set_title('Detected Red Regions')
                axes[0, 1].axis('off')

                # Extracted waveform
                axes[1, 0].plot(df['Time_ms'], df['Amplitude'], 'r-', linewidth=1.5)
                axes[1, 0].set_title('Extracted ECG Signal')
                axes[1, 0].set_xlabel('Time (ms)')
                axes[1, 0].set_ylabel('Amplitude')
                axes[1, 0].grid(True, alpha=0.3)

                # Signal statistics
                axes[1, 1].hist(df['Amplitude'], bins=50, color='skyblue', edgecolor='black')
                axes[1, 1].set_title('Amplitude Distribution')
                axes[1, 1].set_xlabel('Amplitude')
                axes[1, 1].set_ylabel('Frequency')
                axes[1, 1].grid(True, alpha=0.3)

                plt.tight_layout()
                st.pyplot(fig)

                # Download section
                st.markdown("---")
                st.markdown('<h3 class="sub-header">Download Results</h3>', unsafe_allow_html=True)

                # Create Excel file in memory
                excel_buffer = io.BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                    df.to_excel(writer, index=False, sheet_name='ECG_Signal')
                    # Add metadata sheet
                    metadata = pd.DataFrame({
                        'Parameter': ['Source File', 'Total Samples', 'Duration (ms)',
                                      'Min Amplitude', 'Max Amplitude', 'Processing Date'],
                        'Value': [uploaded_file.name, len(df), df['Time_ms'].max(),
                                  df['Amplitude'].min(), df['Amplitude'].max(),
                                  pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')]
                    })
                    metadata.to_excel(writer, index=False, sheet_name='Metadata')

                excel_buffer.seek(0)

                # Download buttons
                col_dl1, col_dl2 = st.columns(2)
                with col_dl1:
                    st.download_button(
                        label="📥 Download Excel File",
                        data=excel_buffer,
                        file_name=f"ecg_signal_{uploaded_file.name.split('.')[0]}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

                with col_dl2:
                    # Option to download as CSV
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download CSV File",
                        data=csv,
                        file_name=f"ecg_signal_{uploaded_file.name.split('.')[0]}.csv",
                        mime="text/csv"
                    )

                # Display success message
                st.success("✅ ECG signal successfully extracted and ready for download!")

        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            st.info("Try adjusting the processing parameters or check if the image meets the requirements.")

else:
    # Display example and instructions when no file is uploaded
    with col2:
        st.markdown('<h3 class="sub-header">How to Use</h3>', unsafe_allow_html=True)

        st.markdown("""
        <div class="info-box">
        <strong>Instructions:</strong>
        1. Upload an ECG image file (PNG, JPG, BMP, or TIFF)
        2. The image should contain:
           - Red ECG waveform
           - Red grid lines/boxes
        3. Adjust processing parameters if needed
        4. Click "Process ECG Image"
        5. Download the extracted signal as Excel/CSV
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### Sample ECG Image Requirements:")

        col_ex1, col_ex2 = st.columns(2)
        with col_ex1:
            st.markdown("**✅ Good Example:**")
            st.markdown("""
            - Clear red waveform
            - Visible grid lines
            - Good contrast
            - No overlapping text
            """)

        with col_ex2:
            st.markdown("**❌ Poor Example:**")
            st.markdown("""
            - Faint waveform
            - Missing grid
            - Multiple colors
            - Heavy noise/artifacts
            """)

        # Example workflow
        st.markdown("### Expected Output Format:")
        example_df = pd.DataFrame({
            'Time_ms': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            'Amplitude': [0.0, 0.12, 0.25, 0.18, 0.05, -0.08, -0.15, -0.10, -0.05, 0.02]
        })
        st.dataframe(example_df, use_container_width=True)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #6c757d; font-size: 0.9rem;'>
    ECG Image to Signal Converter | Built for PhysioNet ECG Dataset | 
    Uses OpenCV, NumPy, SciPy, and Streamlit
    </div>
    """,
    unsafe_allow_html=True
)