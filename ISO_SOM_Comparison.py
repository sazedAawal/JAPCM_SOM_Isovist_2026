import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Load and process image
img = cv2.imread("/Users/sajidiawal/Desktop/JAPCM/Combined_ISO&SOM.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 2. Define colors and masks
som_color = np.array([45, 183, 129])
iso_color = np.array([71, 27, 84])
tol = 40

som_mask = np.all(np.abs(img - som_color) < tol, axis=2)
iso_mask = np.all(np.abs(img - iso_color) < tol, axis=2)

# 3. Apply Morphological Dilation (as per your original code)
kernel = np.ones((5, 5), np.uint8)
som_mask_dilated = cv2.dilate(som_mask.astype(np.uint8), kernel)
iso_mask_dilated = cv2.dilate(iso_mask.astype(np.uint8), kernel)

# 4. Calculate Overlap
overlap = som_mask_dilated & iso_mask_dilated
som_pixels = np.sum(som_mask_dilated)
overlap_pixels = np.sum(overlap)
overlap_percentage = (overlap_pixels / som_pixels) * 100 if som_pixels > 0 else 0

print(f"Overlap Percentage: {overlap_percentage:.2f}%")

# 5. GENERATE LINE CHART DATA
# We sum the pixels along the Y-axis (axis=0) to see distribution across the X-axis
som_line = np.sum(som_mask_dilated, axis=0)
iso_line = np.sum(iso_mask_dilated, axis=0)
overlap_line = np.sum(overlap, axis=0)

# 6. PLOTTING
plt.figure(figsize=(12, 6))

# Plot the SOM and ISO profiles
plt.plot(som_line, label='SOM Density', color='#2DB781', linewidth=2)
plt.plot(iso_line, label='ISO Density', color='#471B54', linewidth=2)

# Highlight the overlap area in the chart to "prove" the intersection
plt.fill_between(range(len(overlap_line)), overlap_line, color='red', alpha=0.3, label='Intersection (Overlap)')

plt.title(f'Spatial Distribution Analysis (Overlap: {overlap_percentage:.2f}%)')
plt.xlabel('Image Width (Pixels)')
plt.ylabel('Pixel Frequency (Column Sum)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()