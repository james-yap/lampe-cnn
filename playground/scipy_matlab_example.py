import scipy.io as sio
import matplotlib.pyplot as plt

# Import MATLAB data using scipy.io
raw_data = sio.loadmat('playground_dataset/Healthy_bulk_data.mat')

# Keys available for use
print(raw_data.keys())

# Shape of image matrix
srs_1450_images = raw_data['Healthy_1450_bgsub']
srs_1668_images = raw_data['Healthy_1668_bgsub']
shg_images = raw_data['Healthy_SHG']
print(shg_images.shape)

# Using matplotlib
fig, axes = plt.subplots(1, 3, figsize=(15,5))

srs_1450 = axes[0].imshow(srs_1450_images[:,:,5], cmap='jet')
axes[0].set_title('SRS 1450 cm$^{-1}$')

srs_1668 = axes[1].imshow(srs_1668_images[:,:,5], cmap='jet')
axes[1].set_title('SRS 1668 cm$^{-1}$')

shg = axes[2].imshow(shg_images[:,:,5], cmap='jet')
axes[2].set_title('SHG')

plt.tight_layout()
plt.show()