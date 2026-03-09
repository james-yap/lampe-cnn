import scipy.io as sio
import matplotlib.pyplot as plt

# Import MATLAB data using scipy.io
raw_data = sio.loadmat('playground_dataset/Healthy_bulk_data.mat')
shg_images = raw_data['Healthy_SHG']
print(shg_images.shape)

# Plot an image as an example
plt.figure(figsize=(6, 6))
plt.imshow(shg_images[:,:,10], cmap='viridis')
plt.colorbar()
plt.title("Healthy (SHG)")
# plt.xlabel("X-axis")
# plt.ylabel("Y-axis")
plt.show()