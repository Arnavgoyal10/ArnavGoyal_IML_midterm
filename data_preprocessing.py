import csv
import pandas as pd

# Define input and output file names
fasta_file = "data/NonAMP"  # Replace with your FASTA file
csv_file = "pipeline/NonAMP.csv"

# Read the FASTA file and process entries
with open(fasta_file, "r") as fasta, open(csv_file, "w", newline="") as csv_out:
    writer = csv.writer(csv_out)
    writer.writerow(["Sequence ID", "Sequence"])  # Write header

    sequence_id = None
    sequence = []

    for line in fasta:
        line = line.strip()
        if line.startswith(">"):  # New sequence ID
            if sequence_id:  # If there's an existing entry, write it
                writer.writerow([sequence_id, "".join(sequence)])
            sequence_id = line[1:]  # Remove '>' from ID
            sequence = []  # Reset sequence
        else:
            sequence.append(line)

    # Write the last sequence
    if sequence_id:
        writer.writerow([sequence_id, "".join(sequence)])

print(f"CSV file saved as {csv_file}")


# Define input and output file names
fasta_file = "data/AMP"  # Replace with your FASTA file
csv_file = "pipeline/AMP.csv"

# Read the FASTA file and process entries
with open(fasta_file, "r") as fasta, open(csv_file, "w", newline="") as csv_out:
    writer = csv.writer(csv_out)
    writer.writerow(["Sequence ID", "Sequence"])  # Write header

    sequence_id = None
    sequence = []

    for line in fasta:
        line = line.strip()
        if line.startswith(">"):  # New sequence ID
            if sequence_id:  # If there's an existing entry, write it
                writer.writerow([sequence_id, "".join(sequence)])
            sequence_id = line[1:]  # Remove '>' from ID
            sequence = []  # Reset sequence
        else:
            sequence.append(line)

    # Write the last sequence
    if sequence_id:
        writer.writerow([sequence_id, "".join(sequence)])

print(f"CSV file saved as {csv_file}")


df1 = pd.read_csv("data/AMPs_protein_properties.csv")
df2 = pd.read_csv("pipeline/AMP.csv")

# Merge the two dataframes
df = pd.merge(df2, df1, on="Sequence ID")
df.drop(columns=["Sequence ID"], inplace=True)  # Drop the sequence ID column
df["amp"] = 1
df.to_csv(
    "pipeline/AMPs_final.csv", index=False
)  # Save the merged dataframe to a CSV file
print("Final CSV file saved as AMPs_final.csv")

df1 = pd.read_csv("data/NonAMPs_protein_properties.csv")
df2 = pd.read_csv("pipeline/NonAMP.csv")

# Merge the two dataframes
df = pd.merge(df2, df1, on="Sequence ID")
df.drop(columns=["Sequence ID"], inplace=True)  # Drop the sequence ID column
df["amp"] = 0
df.to_csv(
    "pipeline/NonAMPs_final.csv", index=False
)  # Save the merged dataframe to a CSV file
print("Final CSV file saved as NonAMPs_final.csv")


df1 = pd.read_csv("pipeline/AMPs_final.csv")
df2 = pd.read_csv("pipeline/NonAMPs_final.csv")
merged_df = pd.concat([df1, df2], ignore_index=True)
merged_df.to_csv("pipeline/final_merged_data.csv", index=False)
print("Merged CSV file saved as merged_data.csv")
