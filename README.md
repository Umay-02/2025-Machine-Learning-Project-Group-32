# 2025-Machine-Learning-Project-Group-32
Group #32 Compliance Radar:
_"As part of a research initiative on corporate integrity and operational risk, our team has been asked to analyze organizational data to uncover patterns that may indicate compliance challenges or ethical inconsistencies across departments. The dataset, org_compliance_data.db, contains diverse information about departments and activities, including operational metrics, reporting frequency, financial indicators, audit results, and anonymized employee engagement scores.
Our task is to design a data-driven analytical framework capable of identifying signals of porential non-compliance, surfaxing key explanatory factors, and supporting evidence-based recommendations for risk mitigation.
The outcome should go beyond prediction, our analysis has combine statistical reasoning, ethical insight, and practical recommendations, enabling the organization to strengthen its internal accountability processes."_

**SECTION 1: OUR INTRODUCTION**
Our project aims to analyze department-level organizational compliance data from an SQLite database by using a complete machine learning pipeline. The main goal is to identify risk patterns across departments and generate interpretable insights using unsupervised learning and exploratory data techniques.

We explored missing datas, cleaned and preprocessed it to create a meaningful composite risk features, to detect outliers and to apply clustering (K-means) to group departments by compliance risk dactor. Visualizations and a recommendation engine help highlight high-risk areas and propose potential improvements.

why we used which libraries?
pandas: to load, clean, transform, and analyze our data.
matplotlib: to visualize and create graphs abour our data.
seabord: this is build on top of matplotlib, we used it to simplify complex plots
scikit-learn: we use it to apply different Machine Learning methods such as StandardScaler, KMeans, and silhouette_score
sqlite3: we used it to access .db database without any extra setup. it enabled us to connect to our dataset org_compliance_data.db

**SECTION 2: METHODS**
We used different methods for different reasons to perfect our model. We used many different libraries such as pandas, matplotlib, seaborn, scikit-learn, sqlite3.
Our database: org_compliance_data.db

**1. Data Extraction:** we loaded SQLite database by using sqlite library. We extracted all tables and saved contents to .txt format
**2. Exploratory Analysis:** We analyzed basic dimensions of the department table and generated many .pdf files to have graphs like histograms, missing value plots, and heatmaps.
**3.  Data Cleaning:** We dropped 30 columns with >40.06% missingness. That particular is from mean of the previous graphs of the missing values -we decided that taking the avarage of missing values among the departments would be a great option to choose a threshold to have a much cleaner outcome. We then imputed mean values for the values of "audit_score_q1", "audit_score_q2", "compliance_score_final"
**4. Feature Engineering:** composite_complianc_score --> average of the 3 compliance features.
risk_index --> 100-composite score --> higher means riskier
**5. Anomaly detection:** We used IQR method to detect outlier departments in terms of risk.
**6. Clustering:** we scaled 4 features using StandardScaler, used Elbow Method to determine K value for K-means (K=3), and then applied K-means clustering and assigned each department to a risk_cluster.
**7. Evaluation:** We evaluated clustering quality using the Silhoutte Score.
**8. Recommendation Engine:** For high-risk outliers, we generated actions recommendations based on conditions like poor audit scores, low engagement, and financial abnormalities.

**SECTION 3: EXPERIMENTAL DESIGN:**
Our objective was to segment departments into meaningful risk groups and validate this segmentation using statistical and visual metrics.

INCLUDED EXPERIMENTS:
-_ IQR-Based Anormally Detection:_ our purpose was to identify extreme risk scores. We took IQR bounds as our metric (1.5x rule), and the outcome was 131 outliers were identified.
- _K-means Clustering:_ Our purpose was to group depertments by similar compliance behaviors. There was no baseline because it was an unsupervised learning; anomaly detection used for validation. Features used during the process: audit_score_q1, audit_score_q2, compliance_score_final, risk_index. evaluation metric: silhoutte score was 0.610 --> reasonable cluster separation

**SECTION 4: RESULTS:**
Our main findings were:
- 709 rows loaded from 'departments' table
- 30 features dropped due to excessive amount of values (above 40.06%)
- Composite scores allowed us to standardize risk
- 131 anomalies detected based on IQR thresholds
- K=3 clusters used


**SECTION 5: CONCLUSION:**
Our project succesfully done and it used a full data pipeline, from raw database to final reporting to cluster departments by compliance risk. Composite scores and visualizations allowed for interpretable segmentation.
K-means offered a reasonable structure, and the Silhouette Score validated its clustering quality. A lightweight reommendation engine further extended the usefulness of the result.





