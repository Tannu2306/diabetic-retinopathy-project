import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

csv_path = "train.csv"
image_folder = "train_images"
output_folder = "dataset"

for split in ["train", "val"]:
    for category in ["No_DR", "DR"]:
        os.makedirs(os.path.join(output_folder, split, category), exist_ok=True)

df = pd.read_csv(csv_path)

df["binary_label"] = df["diagnosis"].apply(lambda x: "No_DR" if x == 0 else "DR")

train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

def copy_images(dataframe, split):
    for _, row in dataframe.iterrows():
        image_name = row["id_code"] + ".png"
        label = row["binary_label"]

        src = os.path.join(image_folder, image_name)
        dst = os.path.join(output_folder, split, label, image_name)

        if os.path.exists(src):
            shutil.copy(src, dst)

copy_images(train_df, "train")
copy_images(val_df, "val")

print("Dataset prepared successfully!")