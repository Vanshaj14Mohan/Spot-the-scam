# 🚨 Spot the Scam - Job Fraud Detection Dashboard

This project is a machine learning-based dashboard that identifies fraudulent job postings using a trained classification model. It takes job listing data from a CSV file, predicts whether each entry is a scam or legitimate, and displays key insights with visualizations.

---

## 📁 Project Structure

```

Spot the scam/
│
├── app/
│   └── dashboard.py              # Streamlit dashboard script
│
├── model/
│   └── scam\_detector\_pipeline.pkl # Trained model (generated after running train\_model.py)
│
├── data/
│   ├── Training Data.csv
│   └── Test Data.csv
│
├── utils/
│   ├── **init**.py
│   ├── model\_utils.py
│   └── prediction.py
│
├── train\_model.py                # Script to train and save model
├── requirements.txt              # Required Python packages
└── README.md                     # Project documentation

```

---

## ⚙️ Setup Instructions

### 1. Clone or Extract the Project

If zipped, extract to a folder like:

```

E:\Spot the scam

````

---

### 2. (Optional but Recommended) Create Virtual Environment

```bash
python -m venv venv
````

Activate it:

```bash
.\venv\Scripts\activate
```

---

### 3. Install Required Libraries

Install dependencies from the provided `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

### 4. Train the Model (if not already trained)

```bash
python train_model.py
```

✅ This will:

* Train a pipeline using `Training Data.csv`
* Save it to `model/scam_detector_pipeline.pkl`
* Show evaluation metrics like accuracy and F1-score

---

### 5. Launch the Streamlit Dashboard

```bash
streamlit run app/dashboard.py
```

🖥 A web browser will open at `http://localhost:8501` where you can:

* Upload a CSV (e.g., `Test Data.csv`)
* View fraud predictions
* Explore fraud probability distribution
* Analyze pie chart of fraud vs legit postings

---

## 📊 Features

* Binary fraud prediction using a trained model pipeline
* Upload and preview top 10 job entries
* Displays fraud probability for each listing
* Pie chart: Legit vs Scam job distribution
* Histogram: Probability distribution of fraud
* Built with `pandas`, `scikit-learn`, `seaborn`, `matplotlib`, and `streamlit`

---

## 📌 Notes

* Input CSVs must contain similar structure and columns as `Training Data.csv`
* Model is trained only on the top 10 rows for demonstration purposes
* For production, re-train with full data

---

## 🧠 Future Enhancements (Optional Ideas)

* Add download button for result CSV
* Add interactive filters (e.g., probability threshold slider)
* Train on full dataset and test with real-world job postings
* Deploy on cloud (Streamlit Cloud, Heroku, etc.)

---

## 👨‍💻 Author

Developed with ❤️ as a part of a Data Science project assignment titled **"Spot the Scam"**.

```
