import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
import xgboost as xgb

df = pd.read_csv("/Users/arjjuns/Downloads/archive (2)/Simulated_Inpatient_Readmission_Dataset.csv")
df.dropna(inplace=True)  # Drop nulls if any

df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1, 'Other': 2})
df['Readmission_Within_30_Days']=df['Readmission_Within_30_Days'].map({'No': 0, 'Yes': 1})

label_encoders = {}
for col in ['Gender', 'Diagnosis_Code']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

plt.figure(figsize=(8, 5))
plt.hist([df[df['Readmission_Within_30_Days'] == 0]['Age'],
          df[df['Readmission_Within_30_Days'] == 1]['Age']],
         bins=20, stacked=True, label=['Not Readmitted', 'Readmitted'], color=['green', 'red'])
plt.title("Age Distribution by Readmission")
plt.xlabel("Age")
plt.ylabel("Count")
plt.legend()
plt.grid(True)
plt.show()

genders = [0, 1, 2]
gender_labels = ['Male', 'Female', 'Other']
not_readmitted = []
readmitted = []

for gender in genders:
    subset = df[df['Gender'] == gender]
    not_readmitted.append((subset['Readmission_Within_30_Days'] == 0).sum())
    readmitted.append((subset['Readmission_Within_30_Days'] == 1).sum())

# Bar plot
x = range(len(genders))
bar_width = 0.4

plt.figure(figsize=(8, 5))
plt.bar(x, not_readmitted, width=bar_width, color='green', label='Not Readmitted')
plt.bar([i + bar_width for i in x], readmitted, width=bar_width, color='red', label='Readmitted')

plt.xticks([i + bar_width / 2 for i in x], gender_labels)
plt.title("Readmission by Gender")
plt.xlabel("Gender")
plt.ylabel("Patient Count")
plt.legend()
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 5))
plt.boxplot([
    df[df['Readmission_Within_30_Days'] == 0]['Risk_Score'],
    df[df['Readmission_Within_30_Days'] == 1]['Risk_Score']
], tick_labels=['Not Readmitted', 'Readmitted'])  # ✅ updated here
plt.title("Risk Score by Readmission Status")
plt.ylabel("Risk Score")
plt.grid(True)
plt.show()

procedures = sorted(df['Number_of_Procedures'].unique())
readmitted_counts = df[df['Readmission_Within_30_Days'] == 1]['Number_of_Procedures'].value_counts().reindex(procedures, fill_value=0)
not_readmitted_counts = df[df['Readmission_Within_30_Days'] == 0]['Number_of_Procedures'].value_counts().reindex(procedures, fill_value=0)

x = range(len(procedures))
bar_width = 0.4

plt.figure(figsize=(8, 5))
plt.bar(x, not_readmitted_counts, width=bar_width, label='Not Readmitted', color='green')
plt.bar([i + bar_width for i in x], readmitted_counts, width=bar_width, label='Readmitted', color='red')

plt.xticks([i + bar_width / 2 for i in x], procedures)
plt.xlabel("Number of Procedures")
plt.ylabel("Patient Count")
plt.title("Procedures vs Readmission")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

X = df.drop(columns=["Patient_ID", "Readmission_Within_30_Days"])
y = df["Readmission_Within_30_Days"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = xgb.XGBClassifier(
    eval_metric='logloss',
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

print(f"Accuracy: {accuracy:.3f}")
print(f"ROC AUC: {roc_auc:.3f}")


fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.title("ROC Curve - XGBoost")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True)
plt.show()

