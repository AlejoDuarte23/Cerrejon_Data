import math
import matplotlib.pyplot as plt
from PIL import Image
import os
import glob
from fpdf import FPDF

output_folder = 'images'
pdf = FPDF(orientation='P', unit='mm', format='A4')
pdf.set_auto_page_break(auto=True, margin=15)

grid_image_file = os.path.join(output_folder, 'grid_image.png')
# Remove the grid image file
try:
# Remove the grid image file
    os.remove(grid_image_file)
except:
    print('yo need to remove the grid_images')

# Get a list of all .png files in the output_folder
image_files = sorted(glob.glob(os.path.join(output_folder, '*.png')))

# Calculate the grid size for subplots
grid_size = math.ceil(math.sqrt(len(image_files)))

fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))


for idx, image_file in enumerate(image_files):
    img = Image.open(image_file)
    ax = axes.flatten()[idx]
    ax.axis('off')
    ax.imshow(img)
    # Add subtitle with image name
    ax.set_title(os.path.splitext(os.path.basename(image_file))[0], fontsize=8)

# Remove unused subplots
for idx in range(len(image_files), grid_size * grid_size):
    axes.flatten()[idx].axis('off')

plt.tight_layout()

# Save the entire grid as a single image
grid_image_file = os.path.join(output_folder, 'grid_image.png')
plt.savefig(grid_image_file, dpi=300)
plt.close(fig)

# Add the grid image to the PDF
img = Image.open(grid_image_file)
width, height = img.size

# Convert image size to millimeters (1 inch = 25.4 mm)
width, height = width * 25.4 / 300, height * 25.4 / 300

# Scale image to fit A4 page
scale = min(pdf.w / width, pdf.h / height)
width, height = int(width * scale), int(height * scale)

pdf.add_page()
pdf.image(grid_image_file, x=(pdf.w - width) / 2, y=(pdf.h - height) / 2, w=width, h=height)



pdf.output('Truck_Location_Report.pdf', 'F')