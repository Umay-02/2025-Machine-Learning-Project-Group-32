# -*- coding: utf-8 -*-
import sqlite3
import os
import pandas as pd
import sqlite3
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


assert os.path.exists("org_compliance_data.db")

conn = sqlite3.connect("org_compliance_data.db")
cursor = conn.cursor()

cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()

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
# Let's first list all available tables in the database to identify the correct one.
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
missing = (departments_df.isnull().mean()*100).sort_values(ascending=False)
missing_nonzero = missing[missing>0]

plt.figure(figsize=(12,8))
plt.barh(missing_nonzero.index, missing_nonzero.values, color="red")
plt.xlabel("% of Mssing values")
plt.title("missing data percentage by feature")
plt.gca().invert_yaxis()
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show

plt.savefig("missing_values_plot.pdf", format="pdf")
plt.show()

print("PDF created successfully!")

'''
PASS
'''



#creating the heatmap (a little complicated...)
plt.figure(figsize=(10,10))
sns.heatmap(departments_df.isnull(),
            cbar=True,
            cmap="Reds",
            yticklabels=True)
plt.title("Heatmap of Missing Values")
plt.tight_layout()
plt.show

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
new_missing_nonzero = new_missing[missing>0]

plt.figure(figsize=(12,8))
plt.barh(new_missing_nonzero.index, new_missing_nonzero.values, color="green")
plt.xlabel("% of Mssing values")
plt.title("missing data percentage by feature after removing")
plt.gca().invert_yaxis()
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show

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

'''
PASS
'''

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

'''
PASS
'''

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

'''
PASS
'''

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
PASS
'''
