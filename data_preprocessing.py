import csv
import pandas as pd

fasta_file = "data/NonAMP"
csv_file = "pipeline/NonAMP.csv"

with open(fasta_file, "r") as fasta, open(csv_file, "w", newline="") as csv_out:
    writer = csv.writer(csv_out)
    writer.writerow(["Sequence ID", "Sequence"])

    sequence_id = None
    sequence = []

    for line in fasta:
        line = line.strip()
        if line.startswith(">"):
            if sequence_id:
                writer.writerow([sequence_id, "".join(sequence)])
            sequence_id = line[1:]
            sequence = []
        else:
            sequence.append(line)

    if sequence_id:
        writer.writerow([sequence_id, "".join(sequence)])

print(f"CSV file saved as {csv_file}")


fasta_file = "data/AMP"
csv_file = "pipeline/AMP.csv"

with open(fasta_file, "r") as fasta, open(csv_file, "w", newline="") as csv_out:
    writer = csv.writer(csv_out)
    writer.writerow(["Sequence ID", "Sequence"])

    sequence_id = None
    sequence = []

    for line in fasta:
        line = line.strip()
        if line.startswith(">"):
            if sequence_id:
                writer.writerow([sequence_id, "".join(sequence)])
            sequence_id = line[1:]
            sequence = []
        else:
            sequence.append(line)

    if sequence_id:
        writer.writerow([sequence_id, "".join(sequence)])

print(f"CSV file saved as {csv_file}")


df1 = pd.read_csv("data/AMPs_protein_properties.csv")
df2 = pd.read_csv("pipeline/AMP.csv")

df = pd.merge(df2, df1, on="Sequence ID")
df.drop(columns=["Sequence ID"], inplace=True)
df["amp"] = 1
df.to_csv("pipeline/AMPs_final.csv", index=False)
print("Final CSV file saved as AMPs_final.csv")

df1 = pd.read_csv("data/NonAMPs_protein_properties.csv")
df2 = pd.read_csv("pipeline/NonAMP.csv")

df = pd.merge(df2, df1, on="Sequence ID")
df.drop(columns=["Sequence ID"], inplace=True)
df["amp"] = 0
df.to_csv("pipeline/NonAMPs_final.csv", index=False)
print("Final CSV file saved as NonAMPs_final.csv")


df1 = pd.read_csv("pipeline/AMPs_final.csv")
df2 = pd.read_csv("pipeline/NonAMPs_final.csv")
merged_df = pd.concat([df1, df2], ignore_index=True)
merged_df.to_csv("pipeline/final_merged_data.csv", index=False)
print("Merged CSV file saved as merged_data.csv")
