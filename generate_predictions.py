import pandas as pd
from predict_resume import predict_resume
from preprocess import clean_resume

def generate_predictions():
    df = pd.read_csv("UpdatedResumeDataSet_Encoded.csv")
    df["cleaned_resume"] = df["Resume"].astype(str).apply(clean_resume)

    df["Predicted_Category"] = df["cleaned_resume"].apply(
        lambda x: predict_resume(x)[0][0]
    )

    df.to_csv("Resume_Predictions.csv", index=False)
    print("Predictions saved.")

if __name__ == "__main__":
    generate_predictions()
