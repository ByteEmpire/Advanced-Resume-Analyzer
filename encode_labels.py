import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset
df = pd.read_csv('UpdatedResumeDataSet.csv')

# Initialize encoder
le = LabelEncoder()

# Encode 'Category' column
df['Category_encoded'] = le.fit_transform(df['Category'])

# Display category mapping
label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print("Category to Number Mapping:\n", label_mapping)

# Save the label mapping to a file
joblib.dump(label_mapping, 'label_mapping.pkl')

print("Label mapping saved successfully!")

# Optional: Save the updated dataset
df.to_csv('UpdatedResumeDataSet_Encoded.csv', index=False)
