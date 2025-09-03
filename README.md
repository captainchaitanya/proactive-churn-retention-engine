# 🚀 Proactive SME Cardholder Retention Engine
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)]((https://proactive-churn-retention-engine-bsbuhzwnddg3w9mlpt3cps.streamlit.app/))

## 🎯 Project Overview

This project is an end-to-end analytical solution designed to proactively identify and mitigate churn among Small-to-Medium Enterprise (SME) cardholders. By leveraging machine learning, the engine predicts which customers are at high risk of inactivity and provides actionable insights and prioritized recommendations for retention teams. This project mirrors the core responsibilities of a data analyst in a marketing, investment, and analytics enablement team, focusing on turning data into tangible business impact.

## ✨ Key Features

- **Churn Prediction:** An XGBoost model trained on transactional data to predict the probability of a customer churning.
- **Precursor Engineering:** Development of key leading indicators of churn, including Recency, Frequency, and Monetary (RFM) scores and other behavioral metrics.
- **Insight Generation:** Use of SHAP (SHapley Additive exPlanations) to explain each prediction, highlighting the specific factors driving a customer's risk.
- **Prioritization Tiers:** Automated segmentation of at-risk customers into Tiers (High, Medium, Low) to help sales and marketing teams prioritize their efforts effectively.
- **Actionable Recommendations:** The dashboard provides customized treatment recommendations for each risk tier, bridging the gap between analysis and business action.
- **Interactive Dashboard:** A user-friendly web application built with Streamlit for real-time analysis and decision support.

## 🛠️ Tech Stack

- **Language:** Python
- **Data Manipulation:** Pandas
- **Machine Learning:** Scikit-learn, XGBoost
- **Model Interpretability:** SHAP
- **Web Framework:** Streamlit
- **Plotting:** Plotly
