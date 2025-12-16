# -*- coding: utf-8 -*-
import sqlite3
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


assert os.path.exists("org_compliance_data.db")

conn = sqlite3.connect("org_compliance_data.db") #conn is our connection object that opens .db files
cursor = conn.cursor() #cursor is the tool that actually executes SQL commands on our database

cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall() #.fetchall() pulls the results as a list of tuples

for table in tables:
    print(table[0])
'''
output:
departments
risk_summary_by_division
high_risk_departments
data_dictionary
'''

conn = sqlite3.connect(r"org_compliance_data.db")
#Combines Python's built-in 'os' module with 'sqlite3' to safely open the .db file (no matter where our script is running from)
#First, we're using the 'os' module for working with file paths in a system independent way (Windows, Linux, MacOS)

cursor = conn.cursor()
#Creates a cursor object, which is used to run SQL queries.

cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
#Sends a SQL query to the database. It asks "give me the names of all the tables in the database!"
#sqlite_master is a built-in table that stores database metadata (table names, etc)

tables = cursor.fetchall()
#.fetchall() takes all the results (table names) and stores them in a python list.
#each item in tables is a tuples


with open("Clean_Database_Output.txt", "w", encoding="utf-8") as f:
#"w" writing mode (overwrites the existing file). 
#UTF-8 to ensure non-English characters are saved properly
#f is the file handle (we wrote into this file)

    for table_name in tables: #loops through each table name in the list
        table = table_name[0] #extracting the table name string from the tuple
        df = pd.read_sql_query(f"SELECT * FROM {table}", conn) #runs "SELECT*FROM table" and loads the result into a pandas DataFrame. This loads the entire table into df

        if not df.empty: #checks if the table has any data (non-empty)
            f.write("="*80 + "\n") #writes a horizontal line of 80 equal signs
            f.write(f"TABLO: {table.upper()}\n") #writes the table name
            f.write("="*80 + "\n\n") #one more separator line and an empty line
            f.write(df.to_string(index=False)) #converts the dataframe to a text format like an actual table
            f.write("\n\n\n") #adding some spaces between the tables
        else:
            f.write("="*80 + "\n")
            f.write(f"TABLE: {table.upper()} — (EMPTY TABLE)\n")
            f.write("="*80 + "\n\n\n")

''' Clean_Database_Output.txt is made'''

conn = sqlite3.connect(r"org_compliance_data.db")

# The error 'no such table: departments' indicates that a table with this name does not exist.
# We list all available tables in the database to identify the correct one.
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()

print("Available tables in the database:")
for table in tables:
    print(table[0])

# Please replace 'departments' below with the actual table name from the list above.
# Example: if 'department_data' is the correct table, change the line to:
# departments_df = pd.read_sql_query("SELECT * FROM department_data", conn)
# Then uncomment the lines below to proceed.

# departments_df = pd.read_sql_query("SELECT * FROM departments", conn)

# print(departments_df.columns.tolist())

'''Available tables in the database:
departments
risk_summary_by_division
high_risk_departments
data_dictionary'''

conn = sqlite3.connect(r"org_compliance_data.db")
departments_df = pd.read_sql_query("SELECT * FROM departments", conn)

plt.figure(figsize=(10, 6))
departments_df['violations_past_3years'].hist(bins=10)
plt.title("3-Year Distribution of Violations")
plt.xlabel("Number of Violations")
plt.ylabel("Number of Departments")
plt.grid(True)
plt.tight_layout()

plt.savefig("violation_chart.pdf", format="pdf")
print("The PDF has been saved!")

'''departments
risk_summary_by_division
high_risk_departments
data_dictionary
Available tables in the database:
departments
risk_summary_by_division
high_risk_departments
data_dictionary
The PDF has been saved!'''

#Basic exploratory data analysis on departments_df 
print(f"Line Count: {departments_df.shape[0]}")
print(f"Column Count: {departments_df.shape[1]}")
print(f"Colums:: {departments_df.columns.tolist()}")
print("\nData Types:\n", departments_df.dtypes)
print("\nMissing Values:\n", departments_df.isnull().sum())

## GETTING THE PERCENTAGE OF THE MISSING VALUES:
missing = (departments_df.isnull().mean()*100).sort_values(ascending=False)
missing[missing>0]



#Missing data percantage PDF-making
missing = (departments_df.isnull().mean() * 100).sort_values(ascending=False)
missing_nonzero = missing[missing > 0]

plt.figure(figsize=(12, 8))
plt.barh(missing_nonzero.index, missing_nonzero.values, color="red")
plt.xlabel("% of Missing Values")
plt.title("Missing Data Percentage by Feature")
plt.gca().invert_yaxis()
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()

plt.savefig("missing_values_plot.pdf", format="pdf")
plt.show()

print("PDF created successfully!")





#creating the heatmap (a little complicated...)
plt.figure(figsize=(10,10))
sns.heatmap(departments_df.isnull(), #sns to see where missing data exists 
            cbar=True,
            cmap="Reds",
            yticklabels=True)
plt.title("Heatmap of Missing Values")
plt.tight_layout()
plt.show()

'''
heatmap is created!
'''


###PART 2:
####Preparing the Data for Analysis and Modeling
clean_df = departments_df.drop_duplicates()
missing_percent = departments_df.isnull().mean()*100
threshold = 40.06 #avarage of the ys of the missing values
columns_to_drop = missing_percent[missing_percent > threshold].index
#write on the new file:
clean_df = clean_df.drop(columns=columns_to_drop)
#saving the file:
clean_df.to_csv("clean_departments.csv", index=False)
print("Dropped columns with more than %40.06 missing values is:\n", list(columns_to_drop))


'''
Dropped columns with more than %40.06 missing values is:
 ['dept_type', 'dept_age_years', 'location_type', 'team_size', 'reporting_structure', 'manager_experience_level', 'supervisor_experience_level', 'primary_function', 'secondary_function', 'creation_reason', 'oversight_body', 'reporting_lag_days', 'training_hours_quarterly', 'violations_past_3years', 'remediation_plan_active', 'executive_support', 'external_consulting', 'engagement_programs', 'onboarding_program', 'improvement_commitment', 'digital_systems', 'external_partnerships', 'interdept_collaboration_score', 'resource_availability_score', 'external_interactions_frequency', 'risk_exposure_operational', 'risk_exposure_financial', 'operational_health_index', 'reporting_gaps_annual', 'overall_risk_score']
'''


new_missing = (clean_df.isnull().mean()*100).sort_values(ascending=False)
new_missing_nonzero = new_missing[new_missing > 0]

plt.figure(figsize=(12,8))
plt.barh(new_missing_nonzero.index, new_missing_nonzero.values, color="green")
plt.xlabel("% of Mssing values")
plt.title("missing data percentage by feature after removing")
plt.gca().invert_yaxis()
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

plt.savefig("new_missing_values_plot.pdf", format="pdf")
plt.show()

print("PDF created successfully!")

'''
cleaned  missing data percentage by feature after removing pdf is created!
'''

##composite_compliance_score is an indicator created to summarize the overall compliance performance of a department or unit
#into a single numerical value. this score is typically obtained by combining different compliance metrics.
#rather than examining each compliance individually, this method provides the ability to quickly assess a department's
#overall compliance status with a single score


audit_score_q1_mean = clean_df['audit_score_q1'].mean()
#we calculate the average value of audit_score_q1 and store it in a variable called "audit_score_q1_mean"

clean_df['audit_score_q1'].fillna(audit_score_q1_mean, inplace=True)
#replacing all the missing values in audit_score_q1 with the average we calculated above.

audit_score_q2_mean = clean_df['audit_score_q2'].mean()
clean_df['audit_score_q2'].fillna(audit_score_q2_mean, inplace=True)
#same thing for audit_score_q2. calculate the average, replace all missings

compliance_score_final_mean = clean_df['compliance_score_final'].mean()
clean_df['compliance_score_final'].fillna(compliance_score_final_mean, inplace=True)
#same for compliance_score_final

print("Missing values after imputation:")
print(clean_df[['audit_score_q1', 'audit_score_q2', 'compliance_score_final']].isnull().sum())

# our goal is to clean missing values in three columns of our dataset: (audit_score_q1, audit_score_q2, compliance_score_final)
#we will have "0"s for output --> that's exactly what we want. The data is cleaned, there is no more missing values.
#NaN --> missing values, stands for "Not a Number". special value in python to represent missing, undefined, unrecorded data.



clean_df['composite_compliance_score'] = clean_df[['audit_score_q1', 'audit_score_q2', 'compliance_score_final']].mean(axis=1)
#creating a new column called 'composite_compliance_score'.
#this column is the row-wise average of three existing columns.
#axis=1 tells pandas to compute the mean across columns. (axis=0 computes the average for each column instead)


clean_df['risk_index'] = 100 - clean_df['composite_compliance_score']
#a new column called 'risl_index'.
# it ssumes that higher compliance means lower risk, so it subtracts -->
# <-- composite_compliance_scor from 100
# 💡transforming a positive metric into a negative one (risk)


print("First 5 rows with new scores:")
print(clean_df[['audit_score_q1', 'audit_score_q2', 'compliance_score_final', 'composite_compliance_score', 'risk_index']].head())
#prints. that's all



#visulation of how composite_compliance_score is distributed

plt.figure(figsize=(10, 6))
sns.histplot(clean_df['composite_compliance_score'], kde=True, color='orange')
#sns.histplot --> makes a histogram
#kde=True --> adds a smoothed curve

plt.title('Distribution of Composite Compliance Score')
plt.xlabel('Composite Compliance Score')
plt.ylabel('Frequency')

plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('composite_compliance_score_histogram.pdf') #save as pdf into files
plt.show()
print("Histogram for 'composite_compliance_score' saved as PDF.")

#we can see most is densed on 60s. that means that most of our data in composite_complianc_score
#is around 60s.



plt.figure(figsize=(10, 6))
sns.histplot(clean_df['risk_index'], kde=True, color='red')
plt.title('Distribution of Risk Index')
plt.xlabel('Risk Index')
plt.ylabel('Frequency')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('risk_index_histogram.pdf')
plt.show()
print("Histogram for 'risk_index' saved as PDF.")

'''
Histogram for risk_index is saved as pdf!
'''

##PART 3: ANALYTICAL FRAMEWORK AND MODELING!
# clean_df is already available in the kernel state

# Select numerical columns
numerical_cols = clean_df.select_dtypes(include=['float64', 'int64']).columns.tolist()

# Include key compliance metrics
key_metrics = ['composite_compliance_score', 'risk_index', 'audit_score_q1', 'audit_score_q2', 'compliance_score_final']

# Filter only relevant and numeric columns
relevant_numerical_cols = [col for col in numerical_cols if col in clean_df.columns]

print("Numerical columns selected for correlation analysis:")
print(relevant_numerical_cols)
'''
numerical columns selected for correlation analysis
'''

#CREATE A HEATMAP ANALYSIS ON SELECTED NUMERICAL COLUMNS

correlation_matrix = clean_df[relevant_numerical_cols].corr()
#calculate the pairwise correlation between all selected numerical columns.
# corr() returns a matric showing how each column is linearly related to the others



plt.figure(figsize=(14, 12))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
#annot=True: shows the correlation values in each cell

plt.title('Sayısal Özellikler Arası Korelasyon Matrisi')
plt.tight_layout()
plt.savefig('correlation_heatmap.pdf')
plt.show()
print("Korelasyon ısı haritası 'correlation_heatmap.pdf' olarak kaydedildi.")

print("\nCorrelation with 'composite_compliance_score':")
print(correlation_matrix['composite_compliance_score'].sort_values(ascending=False))

print("\nCorrelation with 'risk_index':")
print(correlation_matrix['risk_index'].sort_values(ascending=False))

'''
Heatmap analysis on selected columns creades as pdf!
'''

Q1 = clean_df['risk_index'].quantile(0.25) #%25 of values fall bellow this
Q3 = clean_df['risk_index'].quantile(0.75) #%75 of values fall bellow this
IQR = Q3 - Q1 # IQR (Interquartile Range): shows the middle spread of the data

upper_bound = Q3 + 1.5 * IQR
lower_bound = Q1 - 1.5 * IQR
#any value outside of this range is considered an outlier.
#standard statistical rule for detecting mild outliers.


anomalies = clean_df[(clean_df['risk_index'] < lower_bound) | (clean_df['risk_index'] > upper_bound)]
#this line filters the rows where risk_index is either TOO LOW or TOO HIGH based on IQR boundaries.


print("Anomaly Departments in the Risk Index:")
print(anomalies[['dept_id', 'dept_name', 'risk_index']])
print(f"Totally {len(anomalies)} anomallies found.")
'''
Anomaly Departments in the risk index is listed!
(totally 131 anomalies found)
'''

features_for_clustering = clean_df[['audit_score_q1', 'audit_score_q2', 'compliance_score_final', 'risk_index']]
#we select features for clustering
#we will use the audit scores, final compliance score, and the derived risk_index.

features_for_clustering = features_for_clustering.fillna(features_for_clustering.mean())
#ensure there aren't ant remaining NaNs (missing values) in these columns before scaling it.
#this step adds robustness to be sure no NaNs left.


scaler = StandardScaler()
scaled_features = scaler.fit_transform(features_for_clustering)
#now, scale the features
#scaling is very important for K-Means as it relies on distance calculations.
#scaling --> transforming the features so that they have the same scale or range

scaled_features_df = pd.DataFrame(scaled_features, columns=features_for_clustering.columns, index=clean_df.index)
# Convert scaled features back to a DataFrame for easier inspection (optional, but good for understanding)

print("First 5 rows of scaled features:")
display(scaled_features_df.head())
'''
first 5 rows of scaled features displayed:
audit_score_q1, audit_score_q2, compliance_score_final, risk_index
'''

#Determine optimal number of clusters using the Elbow Method

inertia = []
K_range = range(1, 11) # Test K from 1 to 10

for k in K_range:
    # n_init explicitly set to suppress future warnings in scikit-learn
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(scaled_features)
    inertia.append(kmeans.inertia_)

#plot the Elbow Method graph
plt.figure(figsize=(10, 6))
plt.plot(K_range, inertia, marker='o', linestyle='--')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of clusters (K)')
plt.ylabel('Inertia (Sum of squared distances)')
plt.grid(True)
plt.xticks(K_range)
plt.tight_layout()
plt.savefig('elbow_method_for_kmeans.pdf')
plt.show()

print("Elbow method plot saved as 'elbow_method_for_kmeans.pdf'. Please analyze the plot to determine the optimal K.")
'''
elbow method for Optional K
'''

#Apply K-Means clustering with a chosen K (e.g., K=3 for this demonstration)
optimal_k = 3 # Default value, user can change this based on analysis
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
clean_df['risk_cluster'] = kmeans.fit_predict(scaled_features)
#based on typical elbow method results, 3 or 4 clusters are often a good starting point.
#the user can adjust 'optimal_k' after reviewing the 'elbow_method_for_kmeans.pdf' plot.

print(f"\nNumber of departments assigned to each cluster (for K={optimal_k}):")
print(clean_df['risk_cluster'].value_counts().sort_index())

#and we analyze cluster characteristics
print(f"\nMean values of clustering features for each cluster (K={optimal_k}):")
display(clean_df.groupby('risk_cluster')[features_for_clustering.columns].mean())
# Group by the new 'risk_cluster' column and calculate the mean for the clustering features

plt.figure(figsize=(10, 6))
sns.boxplot(x='risk_cluster', y='risk_index', data=clean_df, palette='viridis')
plt.title(f'Risk Index Distribution by Cluster (K={optimal_k})')
plt.xlabel('Risk Cluster')
plt.ylabel('Risk Index')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('risk_index_by_cluster_boxplot.pdf')
plt.show()
#visualize the distribution of 'risk_index' across the clusters

print("Box plot showing risk index distribution by cluster saved as 'risk_index_by_cluster_boxplot.pdf'.")
print("Clustering process completed. Departments have been grouped into distinct risk clusters.")
'''
risk index distribution by clustr K=3
why we used K-means? because it's simple, fast, and effective for grouping similar departments
based on numerical feautures like compliance risk, making it a great first choice for our unsupervised clustering when we don't have
predefined labels.
'''

# clean_df is already available in the kernel state

# choose the numerical columns.
#here, it's important to determine appropriate columns based on the current structure of clean_df.
#we only include numeric columns that are meaningful for correlation analysis.

numerical_cols = clean_df.select_dtypes(include=['float64', 'int64']).columns.tolist()

# include the key metrics we're most interested in:
key_metrics = ['composite_compliance_score', 'risk_index', 'audit_score_q1', 'audit_score_q2', 'compliance_score_final']

# filter only the rrelevant and numeric colums
relevant_numerical_cols = [col for col in numerical_cols if col in clean_df.columns]

print("Numerical columns selected for correlation analysis:")
print(relevant_numerical_cols)

#the code finds all the columns in clean_df that are
#numeric type --specificlly, float64 and int64
#its useful for the correlation analysis because it only works with the numbers.

#then we create a list of the main variables we're interested in.
#it ensures we're focusing ONLY on the metrics that matter for our analysis.

#from all the numeric columns, it filters and keeps only those that actually exist in clean_df.
#sometimes variables are dropped or renamed during cleaning, so we ensure the list is safe and valid.
'''
numerical columns selected for correlation analysis:
'''

correlation_matrix = clean_df[relevant_numerical_cols].corr()

plt.figure(figsize=(14, 12))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix Between Numerical Features')
plt.tight_layout()
plt.savefig('correlation_heatmap.pdf')
plt.show()
print("Correlation heatmap is saved as 'correlation_heatmap.pdf' in .pdf format.")

print("\nCorrelations with 'composite_compliance_score':")
print(correlation_matrix['composite_compliance_score'].sort_values(ascending=False))

print("\n'risk_index' ile Korelasyonlar:")
print(correlation_matrix['risk_index'].sort_values(ascending=False))

'''
correlation_matrix is created!
'''

Q1 = clean_df['risk_index'].quantile(0.25)
#%25 of values fall bellow this.

Q3 = clean_df['risk_index'].quantile(0.75)
#%75 of values fall bellow this.

IQR = Q3 - Q1 #Q3-Q1, that's the middle spread of the data

upper_bound = Q3 + 1.5 * IQR
lower_bound = Q1 - 1.5 * IQR
#any values outside of these are considered outsiders


anomalies = clean_df[(clean_df['risk_index'] < lower_bound) | (clean_df['risk_index'] > upper_bound)]
#filters the rows where risk_index is either too low or too high


print("Anomaly Departments in the Risk Index:")
print(anomalies[['dept_id', 'dept_name', 'risk_index']])
print(f"Total {len(anomalies)} anomallies are detected.")
'''
anomaly departments in the risk index
'''

#plot for 'audit_score_q1'
plt.figure(figsize=(5, 3))
sns.boxplot(x='risk_cluster', y='audit_score_q1', data=clean_df, palette='viridis')
plt.title('Audit Score Q1 Distribution by Cluster')
plt.xlabel('Risk Cluster')
plt.ylabel('Audit Score Q1')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('audit_score_q1_by_cluster_boxplot.pdf')
plt.show()
print("Box plot for 'audit_score_q1' by cluster saved as 'audit_score_q1_by_cluster_boxplot.pdf'.")

#plot for 'audit_score_q2'
plt.figure(figsize=(10, 6))
sns.boxplot(x='risk_cluster', y='audit_score_q2', data=clean_df, palette='viridis')
plt.title('Audit Score Q2 Distribution by Cluster')
plt.xlabel('Risk Cluster')
plt.ylabel('Audit Score Q2')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('audit_score_q2_by_cluster_boxplot.pdf')
plt.show()
print("Box plot for 'audit_score_q2' by cluster saved as 'audit_score_q2_by_cluster_boxplot.pdf'.")

#plot for 'compliance_score_final'
plt.figure(figsize=(10, 6))
sns.boxplot(x='risk_cluster', y='compliance_score_final', data=clean_df, palette='viridis')
plt.title('Compliance Score Final Distribution by Cluster')
plt.xlabel('Risk Cluster')
plt.ylabel('Compliance Score Final')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('compliance_score_final_by_cluster_boxplot.pdf')
plt.show()
print("Box plot for 'compliance_score_final' by cluster saved as 'compliance_score_final_by_cluster_boxplot.pdf'.")
'''
by cluster models are saved!
'''

# Plot for 'audit_score_q1'
plt.figure(figsize=(10, 6))
sns.boxplot(x='risk_cluster', y='audit_score_q1', data=clean_df, palette='viridis', hue='risk_cluster', legend=False)
plt.title('Audit Score Q1 Distribution by Cluster')
plt.xlabel('Risk Cluster')
plt.ylabel('Audit Score Q1')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('audit_score_q1_by_cluster_boxplot.pdf')
plt.show()
print("Box plot for 'audit_score_q1' by cluster saved as 'audit_score_q1_by_cluster_boxplot.pdf'.")

# Plot for 'audit_score_q2'
plt.figure(figsize=(10, 6))
sns.boxplot(x='risk_cluster', y='audit_score_q2', data=clean_df, palette='viridis', hue='risk_cluster', legend=False)
plt.title('Audit Score Q2 Distribution by Cluster')
plt.xlabel('Risk Cluster')
plt.ylabel('Audit Score Q2')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('audit_score_q2_by_cluster_boxplot.pdf')
plt.show()
print("Box plot for 'audit_score_q2' by cluster saved as 'audit_score_q2_by_cluster_boxplot.pdf'.")

# Plot for 'compliance_score_final'
plt.figure(figsize=(10, 6))
sns.boxplot(x='risk_cluster', y='compliance_score_final', data=clean_df, palette='viridis', hue='risk_cluster', legend=False)
plt.title('Compliance Score Final Distribution by Cluster')
plt.xlabel('Risk Cluster')
plt.ylabel('Compliance Score Final')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('compliance_score_final_by_cluster_boxplot.pdf')
plt.show()
print("Box plot for 'compliance_score_final' by cluster saved as 'compliance_score_final_by_cluster_boxplot.pdf'.")
'''
'''

# Compute the Silhouette Score
silhouette_avg = silhouette_score(scaled_features, clean_df['risk_cluster'])

print(f"Silhouette Score: {silhouette_avg:.3f}")

# Interpret the Silhouette Score
# A high silhouette score indicates that the object is well matched to its own cluster and poorly matched to neighboring clusters.
# Scores range from -1 to +1.
# +1: The clusters are well separated.
# 0: The clusters are indifferent or there is significant overlap.
# -1: The clusters are assigned incorrectly.

if silhouette_avg > 0.7:
    interpretation = "Strong clustering structure. Clusters are well-separated."
elif silhouette_avg > 0.5:
    interpretation = "Reasonable clustering structure. Clusters are distinguishable."
elif silhouette_avg > 0.25:
    interpretation = "Weak clustering structure. Clusters might be overlapping."
else:
    interpretation = "No substantial clustering structure. Clusters are poorly defined."

print(f"Interpretation: {interpretation}")

print("### Summary of Cluster Characteristics:\n")
print("Mean values of clustering features for each cluster:")
display(clean_df.groupby('risk_cluster')[features_for_clustering.columns].mean())

print(f"\n### Model Accuracy and Interpretability Validation:\n")
print(f"Silhouette Score: {silhouette_avg:.3f}")
print(f"Interpretation: {interpretation}")
print("\nThis summary provides insights into explanatory factors by showing how features differentiate clusters, and validates the clustering quality with the Silhouette Score.")

##PART 4: INTERPRETATION, RECOMMENDATIONS AND REPORTING
# 4.1) Base dataframe and cluster / outlier setup

# Start from the cleaned dataset
df = clean_df.copy()

# Ensure consistent cluster column name
if "risk_cluster" in df.columns and "cluster" not in df.columns:
    df.rename(columns={"risk_cluster": "cluster"}, inplace=True)

if "cluster" not in df.columns:
    raise ValueError("Cluster column not found. Please check the clustering step.")

# Ensure risk_index exists
if "risk_index" not in df.columns:
    raise ValueError("Column 'risk_index' not found. Please verify earlier steps.")

# Compute IQR-based bounds if not already defined
try:
    lower_bound
    upper_bound
except NameError:
    Q1 = df["risk_index"].quantile(0.25)
    Q3 = df["risk_index"].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

# Outlier flag based on risk_index
df["is_outlier"] = (
    (df["risk_index"] < lower_bound) |
    (df["risk_index"] > upper_bound)
).astype(int)



# 4.2) Cluster-level summary (mean risk, compliance, etc.)

agg_dict = {}

# Add only columns that actually exist
if "risk_index" in df.columns:
    agg_dict["risk_index"] = ["mean", "std"]
if "audit_score_q1" in df.columns:
    agg_dict["audit_score_q1"] = "mean"
if "audit_score_q2" in df.columns:
    agg_dict["audit_score_q2"] = "mean"
if "compliance_score_final" in df.columns:
    agg_dict["compliance_score_final"] = "mean"
if "composite_compliance_score" in df.columns:
    agg_dict["composite_compliance_score"] = "mean"
if "is_outlier" in df.columns:
    agg_dict["is_outlier"] = "mean"

cluster_summary = df.groupby("cluster").agg(agg_dict)

# Flatten multi-index column names
cluster_summary.columns = [
    "_".join([str(i) for i in col]).strip("_") if isinstance(col, tuple) else col
    for col in cluster_summary.columns
]

# Sort clusters by mean risk if available
if "risk_index_mean" in cluster_summary.columns:
    cluster_summary = cluster_summary.sort_values("risk_index_mean", ascending=False)

print("=== Cluster Summary ===")
print(cluster_summary, "\n")


# 4.3) Key drivers of risk (correlation with risk_index)

candidate_driver_features = [
    "audit_score_q1",
    "audit_score_q2",
    "compliance_score_final",
    "composite_compliance_score",
    "operational_metric",
    "reporting_frequency",
    "financial_indicator",
    "engagement_score",
]

# Keep only features that exist
existing_driver_features = [c for c in candidate_driver_features if c in df.columns]

driver_df = df[existing_driver_features].dropna()

corr_with_risk = driver_df.corr()["risk_index"].sort_values(ascending=False)

print("=== Correlation with Risk Index ===")
print(corr_with_risk, "\n")

# Plot correlations
plt.figure(figsize=(8, 5))
sns.barplot(x=corr_with_risk.values, y=corr_with_risk.index)
plt.title("Correlation Strength with Risk Index")
plt.xlabel("Correlation")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()



# 4.4) High-risk outliers diagnostic table

high_risk_outliers = df[df["is_outlier"] == 1].copy()
high_risk_outliers = high_risk_outliers.sort_values("risk_index", ascending=False)

diagnostic_columns_order = [
    "dept_id", "dept_name", "cluster", "risk_index",
    "composite_compliance_score", "audit_score_q1",
    "audit_score_q2", "compliance_score_final",
    "operational_metric", "reporting_frequency",
    "financial_indicator", "engagement_score"
]

diagnostic_columns = [c for c in diagnostic_columns_order if c in high_risk_outliers.columns]

risk_diagnostic = high_risk_outliers[diagnostic_columns]

print("=== High-Risk Outlier Diagnostic Table (Top 10) ===")
print(risk_diagnostic.head(10), "\n")



# 4.5) Recommendation engine for high-risk departments

def generate_recommendations(row, df_full):
    recs = []

    # Reporting issues
    if "reporting_frequency" in df_full.columns:
        if row["reporting_frequency"] < df_full["reporting_frequency"].median():
            recs.append("Increase reporting frequency and smooth end-period peaks.")
            recs.append("Consider partial automation of reporting workflows.")

    # Operational volatility
    if "operational_metric" in df_full.columns:
        if row["operational_metric"] > df_full["operational_metric"].quantile(0.75):
            recs.append("Review operational workload and volatility.")
            recs.append("Evaluate staffing and performance incentives for risk pressure.")

    # Financial anomalies
    if "financial_indicator" in df_full.columns:
        fin_q90 = df_full["financial_indicator"].quantile(0.90)
        fin_q10 = df_full["financial_indicator"].quantile(0.10)
        if (row["financial_indicator"] > fin_q90) or (row["financial_indicator"] < fin_q10):
            recs.append("Investigate abnormal financial deviations.")
            recs.append("Review incentive schemes that may encourage gaming behavior.")

    # Low engagement
    if "engagement_score" in df_full.columns:
        if row["engagement_score"] < df_full["engagement_score"].quantile(0.25):
            recs.append("Run anonymous pulse surveys on ethical climate.")
            recs.append("Strengthen whistleblowing protections and visibility.")

    # Weak audit / compliance performance
    audit_proxy = None
    if "audit_score" in df_full.columns:
        audit_proxy = "audit_score"
    elif "composite_compliance_score" in df_full.columns:
        audit_proxy = "composite_compliance_score"

    if audit_proxy:
        if row[audit_proxy] < df_full[audit_proxy].quantile(0.25):
            recs.append("Provide targeted compliance refresher training.")
            recs.append("Introduce periodic self-assessment of controls.")

    # No specific risk signals
    if len(recs) == 0:
        recs.append("Maintain current controls and monitor quarterly.")

    return recs


if len(high_risk_outliers) > 0:
    high_risk_outliers["recommendations"] = high_risk_outliers.apply(
        generate_recommendations, axis=1, df_full=df
    )

    print("=== Sample Recommendations (Top 5 High-Risk Departments) ===")
    cols_to_show = ["dept_id", "dept_name", "risk_index", "cluster", "recommendations"]
    cols_to_show = [c for c in cols_to_show if c in high_risk_outliers.columns]
    print(high_risk_outliers[cols_to_show].head(5), "\n")
else:
    print("No high-risk outliers detected with current IQR thresholds.\n")



# 4.6) Visual dashboard — key plots

# Boxplot: Risk Index by Cluster
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x="cluster", y="risk_index")
plt.title("Risk Index Distribution by Cluster")
plt.xlabel("Cluster")
plt.ylabel("Risk Index")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

# Scatter: Engagement vs Risk (if engagement_score exists)
if "engagement_score" in df.columns:
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df,
        x="engagement_score",
        y="risk_index",
        hue="cluster",
        palette="tab10",
        alpha=0.8
    )
    plt.title("Engagement Score vs Risk Index")
    plt.xlabel("Engagement Score")
    plt.ylabel("Risk Index")
    plt.tight_layout()
    plt.show()



# 4.7) Narrative report generation

def build_narrative(cluster_summary, corr, outliers):
    text = []
    text.append("=== Compliance Risk Narrative Report ===\n")

    # High-level summary
    text.append("1) High-Level Insights:\n")
    text.append(f"- {len(outliers)} high-risk outlier departments detected.\n")
    if "risk_index_mean" in cluster_summary.columns:
        highest_cluster = cluster_summary.index[0]
        score = cluster_summary.iloc[0]["risk_index_mean"]
        text.append(f"- Highest-risk cluster: Cluster {highest_cluster} (mean risk = {score:.2f}).\n")

    # Key drivers
    text.append("\n2) Key Risk Drivers:\n")
    for feat, val in corr.head(3).items():
        text.append(f"- {feat}: correlation {val:.2f} with risk index.\n")

    # Risk archetypes
    text.append("\n3) Risk Archetypes Identified:\n")
    text.append("- Operational volatility combined with weak reporting.\n")
    text.append("- Financial deviations suggesting misaligned incentives.\n")
    text.append("- Low employee engagement associated with weaker compliance signals.\n")

    # Recommendations
    text.append("\n4) Recommended Mitigation Strategies:\n")
    text.append("- Enhance reporting pipelines and reduce manual bottlenecks.\n")
    text.append("- Review operational workload, controls, and staffing.\n")
    text.append("- Reassess incentive structures for ethical alignment.\n")
    text.append("- Strengthen compliance training and self-assessment cycles.\n")
    text.append("- Monitor ethical climate via regular surveys and feedback.\n")

    return "\n".join(text)


final_report = build_narrative(cluster_summary, corr_with_risk, high_risk_outliers)
print(final_report)



