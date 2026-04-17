"""
Generate LinkedIn carousel PDF from dashboard screenshots.
Each image becomes a full-bleed page in landscape orientation.
"""
from PIL import Image
import os

CAROUSEL_DIR = r"c:\Users\Lenovo\Design Thinking And Innovation Project\dashboard\linkedin_carousel"
OUTPUT_PDF = r"c:\Users\Lenovo\Design Thinking And Innovation Project\dashboard\EquiChurn_LinkedIn_Carousel.pdf"

# Get all slides in order
slides = sorted([
    os.path.join(CAROUSEL_DIR, f) 
    for f in os.listdir(CAROUSEL_DIR) 
    if f.endswith('.png')
])

print("Found %d slides:" % len(slides))
for s in slides:
    print("  - %s" % os.path.basename(s))

# Open all images and convert to RGB (PDF doesn't support RGBA)
images = []
for slide_path in slides:
    img = Image.open(slide_path).convert('RGB')
    images.append(img)

# Save as multi-page PDF
if images:
    first_image = images[0]
    remaining = images[1:]
    
    first_image.save(
        OUTPUT_PDF,
        "PDF",
        resolution=150.0,
        save_all=True,
        append_images=remaining
    )
    
    file_size_mb = os.path.getsize(OUTPUT_PDF) / (1024 * 1024)
    print("")
    print("PDF created successfully!")
    print("Location: %s" % OUTPUT_PDF)
    print("Pages: %d" % len(images))
    print("Size: %.1f MB" % file_size_mb)
else:
    print("No images found!")
