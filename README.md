# ECG Converter - Electrocardiogram Signal Converter

A **Streamlit-based ECG Converter** that allows users to upload ECG signal files and process them into visual plots and data formats for analysis. This tool is helpful for biomedical engineers, students, and researchers working with ECG signals.  

---

## 🚀 Features

- Upload ECG signals in supported formats.
- Visualize ECG waveforms with **interactive plots**.
- Convert ECG data into **structured formats** for analysis.
- Simple and clean **Streamlit interface** for easy usage.
- Handles multiple signals and large datasets efficiently.

---

## 💻 Technologies Used

- **Python 3.x**  
- **Streamlit** for the web interface  
- **OpenCV (opencv-python-headless)** for image processing  
- **NumPy & Pandas** for numerical and tabular data  
- **SciPy** for signal interpolation  
- **Matplotlib** for plotting ECG waveforms  
- **Pillow** for image handling  

---

## 📂 Installation

1. Clone this repository:
```bash
git clone https://github.com/junaid1233/ECG_-Electrocardiogram-_Converter.git
cd ECG_-Electrocardiogram-_Converter
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
pip install -r requirements.txt
streamlit run ecg_convertor.py

