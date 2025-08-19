# Hospital Readmission Prediction

This project predicts whether a patient is likely to be readmitted to the hospital using machine learning.

## Project Structure
├── app.py # Backend model (training + prediction logic)

├── frontend.py # Streamlit frontend for user interaction

├── requirements.txt

├── README.md

## ⚙️ Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/hospital-readmission.git
   cd hospital-readmission

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # on Mac/Linux
   venv\Scripts\activate     # on Windows

3.Install dependencies:  
    ```bash   
    pip install -r requirements.txt

## Running the Project
1. Run the backend (model training & saving)
    ```bash
    python app.py
2. Run the Streamlit frontend
    ```bash
    streamlit run frontend.py
Then open the link provided by Streamlit (usually http://localhost:8501).

## Features

User-friendly UI with Streamlit
Predicts hospital readmission (Yes/No)
Shows probability score
Easily deployable on GitHub + Streamlit Cloud / HuggingFace Spaces
