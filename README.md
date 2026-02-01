# Project 1 - Python 2

**Group members:** Ryan NAIDJI, Abdelkrim INNOUCHE

project1_retail_analytics/
│
├── data/
│ └── retail_orders_large.csv   # dataset
│
├── notebooks/
│ └── Project1_Retail_Exploration.ipynb # notebook
│
├── src/
│ ├── revenue_analysis.py       # top-k algorithm, rolling avg window algorithm
│ ├── anomaly_detection.py      # z-score, duplicate rows
│ └── utils.py                  # utilities
│
├── outputs/
│ └── screenshots/
│
├── README.md
├── proposal.pdf                # project proposal
└── requirements.txt            # python dependencies

## How to Run

### 1) Create and activate a virtual environment

```bash
python -m venv venv
venv/Scripts/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Open the notebook in `notebooks/` and run it