import glob
import math
import os
from PIL import Image
from fpdf import FPDF
import matplotlib.pyplot as plt

# Get a list of all .svg files in the images folder
image_files = sorted(glob.glob(os.path.join('images', '*.png')))

# Calculate the grid size for subplots (2 columns)
num_rows = math.ceil(len(image_files) / 2)

fig, axes = plt.subplots(num_rows, 2, figsize=(10, 10 * num_rows))

# Display images in the subplots
for idx, image_file in enumerate(image_files):
    img = Image.open(image_file)
    ax = axes.flatten()[idx]
    ax.axis('off')
    ax.imshow(img)
    ax.set_title(os.path.splitext(os.path.basename(image_file))[0], fontsize=8)

# Remove unused subplots
for idx in range(len(image_files), num_rows * 2):
    axes.flatten()[idx].axis('off')

plt.tight_layout()

# Save the entire grid as a single PNG image
grid_image_file = os.path.join('images', 'grid_image.png')
plt.savefig(grid_image_file, dpi=300)
plt.close(fig)

# Create a PDF and add the grid image
pdf = FPDF(orientation='P', unit='mm', format='A4')
pdf.set_auto_page_break(auto=True, margin=15)

img = Image.open(grid_image_file)
width, height = img.size

# Convert image size to millimeters (1 inch = 25.4 mm)
width, height = width * 25.4 / 300, height * 25.4 / 300

# Scale image to fit A4 page
scale = min(pdf.w / width, pdf.h / height)
width, height = int(width * scale), int(height * scale)

pdf.add_page()
pdf.image(grid_image_file, x=(pdf.w - width) / 2, y=(pdf.h - height) / 2, w=width, h=height)

# Save the PDF
pdf.output('Truck_Location_Report.pdf', 'F')
