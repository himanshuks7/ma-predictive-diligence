# ma-predictive-diligence
M&amp;A Predictive Diligence Engine with Live AI Analysis
# M&A Predictive Diligence & Regulatory Risk Engine

A predictive analytics and generative AI pipeline built with Scikit-Learn (Random Forest), Pandas, Streamlit, and the Google Gemini API. 

### 1) Project Goal

The project simulates and predicts the probability of regulatory blocks for proposed corporate acquisitions using a machine learning model trained on historical M&A datasets. It evaluates market concentration dynamics and leverages Generative AI (Gemini 3.5 Flash) to automatically draft professional institutional risk memorandums based on live ML inference outputs.

### 2) Data Flow (End-to-End)

1. `pipeline.py` ingests historical M&A data from the raw CSV dataset.
2. The pipeline computes a `Sector_Congestion_Index` for each industry, imputes missing values, encodes categories, and exports the clean data.
3. `app.py` loads the clean dataset and dynamically trains a Scikit-Learn `RandomForestClassifier` on startup.
4. Users input live deal parameters (Acquisition Price, Target Sector) via the Streamlit UI.
5. The model instantly calculates the probabilistic risk of a regulatory block vs. successful completion.
6. The user clicks "Generate Live AI Memo", triggering the Gemini API to evaluate the quantitative risk output and synthesize a 3-sentence investment committee memo.
7. If API limits are reached, a dynamic heuristic fallback engine automatically renders a simulated memo.

### 3) Relevant Files

**Data Engineering Pipeline**
* `pipeline.py`
  * **Purpose:** Safely ingests raw corporate M&A datasets and engineers features necessary for ML training.
  * **Key details:** 
    * Calculates sector congestion proxies.
    * Encodes target variables (`Deal_Status`).
    * Outputs to: `data/cleaned_ma_data.csv`.

**Dashboard & AI Engine**
* `app.py`
  * **Purpose:** Interactive Streamlit UI for real-time risk inference and AI synthesis.
  * **Modes:**
    * **Live Deal Parameters:** Takes continuous user input for deal size and sector.
    * **Main Dashboard Metrics:** Displays quantitative risk probabilities and system latency.
    * **AI Risk Memo Generator:** Connects to `gemini-3.5-flash` via the `google-genai` SDK to draft contextual text.
    * **Historical Precedent:** Uses Pandas structural matching to display the top 5 largest historical transactions in the selected sector.

### 4) Data Folder

* `data/acquisitions_update_2021.csv`
  * Main historical dataset used for initial ingestion.
* `data/cleaned_ma_data.csv`
  * The processed output from the pipeline used for dynamic model training and precedent matching in the dashboard.
  * Expected columns referenced by code include: `Acquisition Price`, `Sector_Congestion_Index`, `Category`, `Parent Company`, `Acquired Company`, and `Deal_Status`.

### 5) Prerequisites

Install and run the following locally:
* Python 3.9+
* Required libraries: `streamlit`, `pandas`, `numpy`, `scikit-learn`, `google-genai`, `python-dotenv`

