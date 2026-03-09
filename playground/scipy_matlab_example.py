import scipy.io as sio

data = sio.loadmat('playground_dataset/Healthy_bulk_data.mat')
print(data['Healthy_SHG'].shape)