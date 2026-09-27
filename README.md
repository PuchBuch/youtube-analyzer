# 📊 YouTube Competitor Intelligence & ML Performance Analyzer

An end-to-end data engineering and machine learning pipeline that extracts real-world channel metrics via the **YouTube Data API v3** and uses a **Random Forest Classifier** to reverse-engineer and predict video performance patterns.

---

## 🚀 Business Value & Core Features

- **Automated Data Harvesting:** Safely queries and structures deep metadata (views, velocity per day, likes, comments, precise video duration, tags, description attributes, and channel subscriber counts).
- **Predictive ML Modeling:** Trains an ensemble model (`RandomForestClassifier`) to differentiate high-performing videos (top 25% by daily view growth) and calculates exact feature importance.
- **Executive Dashboard Generation:** Automatically outputs an isolated, modern HTML/CSS reporting dashboard displaying key performance indicators, Top 10 viral videos, and data-driven content strategy insights.

---

## 🛠 Tech Stack & Dependencies

```text
Python 3.11
├── google-api-python-client  — YouTube API v3 Integration
├── pandas & numpy            — Data manipulation & feature engineering
├── scikit-learn              — Train-test split & RandomForest modeling
└── python-dotenv             — Secure environment configuration
```

---

## 📂 Project Architecture & Deliverables

- `youtube_real.py` — The core script executing API extraction, ML modeling, and reporting.
- `youtube_real_data.csv` — The generated local dataset containing structured raw video metrics.
- `youtube_real_report.html` — The dynamic client-ready dashboard with visualized metrics.

---

## 📊 Visualizing Results (Dashboard Preview)

*Tip: Open `youtube_real_report.html` in your browser to inspect the interactive report.*

### 1. Key Performance Indicators (KPIs)
The system establishes a clear baseline for data validation:
- **Success Threshold:** Calculated dynamically based on view velocity per day.
- **Model Accuracy:** Evaluation score demonstrating high predictive reliability.

### 2. Feature Importance Matrix
The Machine Learning model ranks features by their statistical impact on a video's success:
1. **Channel Subscribers** (Primary driver for initial velocity)
2. **Video Duration** (Optimal window identified at 10–15 minutes)
3. **Title Tags & Hooks** (Presence of "How to" and numerical list triggers)

---

## ⚙️ Installation & Local Setup

### 1. Clone the repository
```bash
git clone https://github.com
cd youtube-analyzer
```

### 2. Activate Virtual Environment
**Windows (PowerShell):**
```powershell
.\venv\Scripts\Activate.ps1
```
**Linux / macOS:**
```bash
source venv/bin/bin/activate
```

### 3. Install Dependencies
```bash
pip install google-api-python-client pandas numpy scikit-learn python-dotenv
```

### 4. Configure Environment Variables
Create a `.env` file in the root directory:
```env
YOUTUBE_API_KEY=your_actual_youtube_api_key_here
```

### 5. Run the Analyzer
```bash
python youtube_real.py
```

---

## 📝 Developer Profile & Freelance Inquiries

Developed as part of a high-performance Python automated solutions portfolio. Available for contract work regarding:
- **Custom API Integrations & Data Engineering Pipelines**
- **Web Scraping & Browser Automation (Selenium / Stealth Technology)**
- **Business Intelligence Dashboards & Predictive Analytics**

📩 **Upwork Profile:** [Upwork Freelancer Profile](https://www.upwork.com/freelancers/~01fd4904f07a135f68)
