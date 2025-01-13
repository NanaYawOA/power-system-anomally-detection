
import os
import pandas as pd 
import numpy as np

# Specify the directory containing the CSV files
csv_directory = "C:/Users/Hp/Desktop/code_projects/power-system-anomally-detection/datasets/triple"  # Replace with your folder path

# List all CSV files in the directory
csv_files = [file for file in os.listdir(csv_directory) if file.endswith('.csv')]

# Create an empty list to store dataframes
dataframes = []

# Loop through each CSV file and read it into a dataframe
for csv_file in csv_files:
    file_path = os.path.join(csv_directory, csv_file)
    df = pd.read_csv(file_path)  # Read the CSV file
    dataframes.append(df)  # Add the dataframe to the list

# Combine all dataframes into one
combined_df = pd.concat(dataframes, ignore_index=True)

# Replace infinite values with NaN and drop rows with NaN values
combined_df.replace([np.inf, -np.inf], np.nan, inplace=True)
combined_df.dropna(inplace=True)


# Save the combined dataframe to a new CSV file
output_path = os.path.join(csv_directory, "powersys_data_supervised_triple.csv")
combined_df.to_csv(output_path, index=False)

print(f"Combined CSV saved to {output_path}")