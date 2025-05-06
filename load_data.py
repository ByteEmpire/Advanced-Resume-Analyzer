import pandas as pd

# Load the dataset
df = pd.read_csv('UpdatedResumeDataSet.csv')

# Show the first few rows of the dataset
print(df.head())

# Display basic information about the dataset
print(df.info())
