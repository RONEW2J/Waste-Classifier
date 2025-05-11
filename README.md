# Waste Classification Project Website

## How to Run Locally
1. Install Python requirements:
```bash
pip install -r ../requirements.txt
```

2. Run the Gradio app:
```bash
python ../app.py
```

3. Open `index.html` in any browser

## Live Demo
[![Gradio Demo](https://img.shields.io/badge/Gradio-Live_Demo-blue)](https://your-gradio-app-url)

## Project Structure
```
data/
├── dataset-resized/  # TrashNet data
└── custom_data/      # Our collected images
models/               # Trained models
website/
├── index.html        # This website
├── README.md
└── assets/           # Images and graphs
```