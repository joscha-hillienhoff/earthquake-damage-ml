# 🌍 Earthquake Damage Prediction (ML)

This project predicts **building damage grades** caused by the **2015 Gorkha earthquake in Nepal** using machine learning.  
It builds on the [DrivenData “Richter’s Predictor” competition](https://www.drivendata.org/competitions/57/nepal-earthquake/) and implements an end-to-end ML workflow with **Bayesian hyperparameter optimization**, **MLflow tracking**, and **DagsHub integration**.

---

## 📝 Project Description

On April 25, 2015, a **7.6 magnitude earthquake** struck the Gorkha region of Nepal, causing devastating destruction and economic losses estimated at $7 billion.  
To support **risk assessment** and **disaster preparedness**, this project develops models to **predict the damage grade** of buildings based on structural and geospatial features.

The workflow follows the **CRISP-DM** methodology:
1. Business Understanding  
2. Data Understanding  
3. Data Preparation  
4. Modeling  
5. Evaluation  
6. Deployment  

### ✨ Key Features
- Data preprocessing and feature engineering
- **Random Forest** and **Extreme Gradient Boosting (XGBoost)** models
- **Bayesian optimization** using `skopt` for hyperparameter tuning
- Evaluation with **Micro-averaged F1 Score**
- Automated experiment logging with **MLflow** + DagsHub
- Feature importance visualization

### 🧰 Technologies Used
- Python 3.11, scikit-learn, XGBoost, skopt  
- MLflow, DagsHub (experiment tracking)  
- Hydra (configuration), Typer (CLI)  
- Poetry (dependency management)

### 🚀 Challenges & Outlook
- Managing imbalanced classes (e.g., `damage_grade = 2` dominates)
- Ensuring feature alignment after one-hot encoding
- Balancing model performance vs. interpretability

---

## 🧭 Project Structure
The project uses the Cookiecutter Data Science v2 template.

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         earthquake_damage_ml and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── src   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes earthquake_damage_ml a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

## ⚙️ Installation

Follow these steps to set up the project locally:

### 1. Clone the Repository
```bash
git clone https://github.com/joscha0610/earthquake-damage-ml.git
cd earthquake-damage-ml
```
### 2. Create Poetry environment
```bash
make create_environment
```
### 3. Install project dependencies
```bash
make requirements
```

## 📊 Results

| Model | Micro-F1 (Val) | Micro-F1 (Test) | Rank (Competition) |
| :--- | :---: | :---: | :---: |
| Random Forest (tuned) | 0.739 | 0.7397 | 963 / 2258 |
| **XGBoost (tuned)** | **0.744** | **0.7444** | **710 / 2258** |

***

| Item | Detail |
| :--- | :--- |
| **✅ Best Model** | **XGBoost** with Bayesian hyperparameter tuning |
| **🏗 Key Features** | `geo_level_*_id`, structural attributes, foundation and roof type |
| **📏 Evaluation Metric** | **Micro-averaged F1** (DrivenData standard) |
