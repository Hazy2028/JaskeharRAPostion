from PIL import Image
import numpy as np
import os

# Directory containing the original black-and-white images
input_dir = '/uufs/chpc.utah.edu/common/home/u1313462'
# Directory where the converted images will be saved
output_dir = '/uufs/chpc.utah.edu/common/home/u1313462/VQA-Med-2019/VQAMed2019Test/ParentImagesClass/Train_images'

# Make sure the output directory exists, create if it doesn't
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Iterate over all files in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):  # Check for image files
        # Construct the full file path
        file_path = os.path.join(input_dir, filename)
        
        # Load the black-and-white image and convert to grayscale ('L')
        bw_image = Image.open(file_path).convert('L')
        
        # Convert the grayscale image to RGB
        rgb_image = Image.merge("RGB", (bw_image, bw_image, bw_image))
        
        # Resize the image to the required size (224, 224) if necessary
        required_size = (224, 224)
        rgb_image = rgb_image.resize(required_size)
        
        # Construct the output file path
        output_file_path = os.path.join(output_dir, filename)
        
        # Save the converted image
        rgb_image.save(output_file_path)
        print(f"Converted and saved: {filename}")

print("Conversion completed for all images.")