# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 13:05:46 2024

@author: TEJA
"""

import os
import csv

# Define directory containing subdirectories
root_dir = "E:/programmer me/unlearning/datasets/cplfw_aligned"

# Define output CSV file
csv_file = "lfw.csv"

csv_data = []

# Traverse through the directory
serial_number = 1
for class_folder in os.listdir(root_dir):
    class_path = os.path.join(root_dir, class_folder)
    if os.path.isdir(class_path):
        class_number = class_folder
        for image_file in os.listdir(class_path):
            if image_file.endswith('.jpg') or image_file.endswith('.png'):  # Adjust file extensions as needed
                image_id = image_file.split('.')[0] 
                csv_data.append([serial_number, str(image_id), class_number,'.jpg'])
                serial_number += 1
                
# Write CSV data to file
with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['id', 'name', 'class','ext'])
    for row in csv_data:
        writer.writerow([row[0],row[1], row[2],row[3]])

#%%
import chardet

def detect_encoding(file_path):
  """
  Detects the encoding of a text file using chardet library.

  Args:
      file_path (str): Path to the text file.

  Returns:
      dict: Dictionary containing the detected encoding and confidence level.
  """
  with open(file_path, 'rb') as f:
    rawdata = f.read()

  return chardet.detect(rawdata)

# Example usage
file_path = "files/lfwd.csv"  # Replace with your file path
encoding_info = detect_encoding(file_path)

if encoding_info:
  print(f"Detected encoding: {encoding_info['encoding']} (confidence: {encoding_info['confidence']:.2f})")
else:
  print("Unable to detect encoding with certainty.")


#%%


import os

def group_photos_by_name(directory):
  """
  Groups photos in the specified directory into subdirectories based on the name in the filename (firstname_lastname_id format).

  Args:
    directory: The directory containing the photos.
  """
  for filename in os.listdir(directory):
    if filename.endswith(".jpg") or filename.endswith(".png"):  # Adjust extensions as needed
      # Extract name from filename (assuming format "firstname_lastname_id")
      name_parts = filename.split("_")[:-1]  # Exclude the last part (id)
      name = " ".join(name_parts)

      # Create subdirectory if it doesn't exist
      person_dir = os.path.join(directory, name)
      if not os.path.exists(person_dir):
        os.makedirs(person_dir)

      # Move the photo to the subdirectory
      source = os.path.join(directory, filename)
      destination = os.path.join(person_dir, filename)
      os.rename(source, destination)

# Replace 'path/to/your/directory' with the actual directory path containing the photos
#group_photos_by_name('E:/programmer me/unlearning/datasets/Face Recognition/Detected Faces')

group_photos_by_name('E:/programmer me/unlearning/datasets/cplfw/images')


#%%

# Open the CSV file for writing
with open(output_file, 'w', newline='') as csvfile:
    # Create a CSV writer object
    writer = csv.writer(csvfile)

    # Write the header row
    writer.writerow(["id", "name", "class", "ext"])

    # Iterate over all subdirectories
    for root, dirs, files in os.walk(data_dir):
        for directory in dirs:
            # Extract information
            name = directory
            class_id = name

            # Assuming only image files with known extensions
            for file in files:
                # Extract extension
                ext = file.split('.')[-1]

                # Write data to CSV
                writer.writerow([name, name, class_id, ext])

print(f"CSV file created: {output_file}")
