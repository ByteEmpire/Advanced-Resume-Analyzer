import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Load the cleaned data
df = pd.read_csv('UpdatedResumeDataSet.csv')

# If you previously saved 'cleaned_resume', use that; otherwise apply quick cleaning here
if 'cleaned_resume' not in df.columns:
    df['cleaned_resume'] = df['Resume']

# Combine all cleaned resumes into one text
text = ' '.join(df['cleaned_resume'])

# Generate the word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

# Plot the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud of Resume Texts")
plt.show()
