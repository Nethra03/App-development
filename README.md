# Food Recognition App (Indian Foods)

## Structure
- frontend/ → HTML, CSS, JS for UI
- backend/ → Flask API + trained model
- training/ → Train model on Food-101 (Indian subset)
- requirements.txt

## Usage
1. Install dependencies: `pip install -r requirements.txt`
2. Train the model: `python training/train_food101_indian.py`
3. Start backend: `python backend/app.py`
4. Open `frontend/index.html` in browser.
