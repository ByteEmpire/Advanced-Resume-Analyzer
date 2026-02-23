import pandas as pd
import joblib

df = pd.read_csv("UpdatedResumeDataSet.csv")

categories = sorted(df["Category"].unique())
label_to_id = {c: i for i, c in enumerate(categories)}
id_to_label = {i: c for c, i in label_to_id.items()}

df["Category_encoded"] = df["Category"].map(label_to_id)

joblib.dump(
    {"label_to_id": label_to_id, "id_to_label": id_to_label},
    "label_mapping.pkl"
)

df.to_csv("UpdatedResumeDataSet_Encoded.csv", index=False)

print("Label encoding completed.")
