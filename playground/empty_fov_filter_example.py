import scipy.io as sio
import matplotlib.pyplot as plt

all_3x3 = sio.loadmat('lampe_dataset/3x3 all images/IDC_bulk_data.mat')
filtered_3x3 = sio.loadmat('lampe_dataset/3x3 bad SHG removed/IDC_bulk_data.mat')

print(all_3x3.keys())
print(all_3x3['IDC_1668_onres'].shape, filtered_3x3['IDC_1668_onres'].shape)

modalities = ['IDC_1450_onres', 'IDC_1668_onres', 'IDC_SHG', 'IDC_1450_bgsub', 'IDC_1668_bgsub']

filtered_set = set()
for img in filtered_3x3['IDC_SHG'].transpose(2,0,1):
    filtered_set.add(img.tobytes())

idx = 0
for i, img in enumerate(all_3x3['IDC_SHG'].transpose(2,0,1)):
    if img.tobytes() not in filtered_set:
        fig, ax = plt.subplots(2, 3, figsize=(15,10))
        
        for j in range(len(modalities)):
            row, col = divmod(j, 3)
            
            mode = modalities[j]

            ax[row, col].imshow(all_3x3[mode][:,:,i], cmap='jet')
            ax[row, col].set_title(f'{mode} ({i})', fontsize=10)
            ax[row, col].axis('off')

        fig.savefig(f'playground/empty_fov/empty_fov_{idx}_{i}.png', bbox_inches='tight', dpi=150)
        plt.close(fig) # Close the figure to free up memory after saving
        idx += 1

assert idx == all_3x3['IDC_1668_bgsub'].shape[2] - filtered_3x3['IDC_1668_bgsub'].shape[2], \
"Number of empty FOVs does not match the expected count."