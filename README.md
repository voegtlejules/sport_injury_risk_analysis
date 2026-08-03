# Explainable Machine Learning for Sports Injury Risk Analysis

**A Proof-of-Concept Study for Explainable AI (XAI) in Sports Medicine**

This repository contains the source code for an interactive decision-support web application developed during a research project at the **Statistical Lab, Graduate School of Information Sciences, Tohoku University** (COLABS Program). 

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://sport-injury-risk-analysis.streamlit.app/)

## Project Overview
Professional sports have seen a marked increase in the number of injuries due to congested schedules and intensified gameplay. While Machine Learning (ML) models offer promising predictive capabilities for injury risk, their clinical adoption is limited because they encountered the "Black Box" problem. Medical staff require actionable insights, not just binary classifications.

This Proof of Concept (PoC) aims to bridge the gap between advanced ML and practical injury risk analysis. By comparing a baseline **Logistic Regression** model with a complex **Random Forest** classifier, this project utilizes advanced **Explainable AI (XAI)** techniques to extract transparent, actionable clinical insights from physiological and lifestyle data.

## Key Features
*   **Model Comparison**: Evaluates and interprets a Regularized Logistic Regression alongside a Random Forest classifier.
*   **Global Interpretability**: Utilizes Odds Ratios, Permutation Feature Importance (PFI), Partial Dependence Plots (PDP), and Individual Conditional Expectation (ICE) to understand team-wide risk trends.
*   **Individual Interpretability**: Leverages **SHAP (SHapley Additive exPlanations)** values to provide actionable, player-specific clinical diagnoses (e.g., quantifying the exact probability shift associated with a player's sleep schedule or nutrition quality).
*   **Team-Wide Risk Assessment**: Allows medical staff to upload a CSV file containing team data to identify macro-trends and establish targeted, preventative training programs.
*   **Nuanced Risk Classification**: Replaces simplistic binary classification (At Risk / Not At Risk) with a 5-tier clinical risk grid (Very Low to Critical) for tailored medical interventions.

## Repository Structure
*   `app.py`: The main Streamlit application script containing the UI and logic.
*   `models/`: Serialized pre-trained models (`.pkl`) to ensure real-time fluidity on the web interface.
*   `ICE/`: Pre-computed Partial Dependence and ICE plots for rapid visualization.
*   `data.csv`: A sample synthetic dataset representing players competing in the Chinese University Championship.
*   `requirements.txt`: Python dependencies required to run the application.

## How to Run Locally

1. **Clone the repository**
   git clone [https://github.com/voegtlejules/sport_injury_risk_analysis.git](https://github.com/voegtlejules/sport_injury_risk_analysis.git)
   cd sport_injury_risk_analysis
2. ## Install Dependencies
   pip install -r requirements.txt
3. ## Run the Streamlit application
   streamlit run app.py

## Author 
Jules Voegtlé
1st Year Master Student, INSA Toulouse / Tohoku University (COLABS Program)
Contact: voegtlejules@gmail.com
