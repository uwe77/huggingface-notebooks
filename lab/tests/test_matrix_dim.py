import numpy as np

# Creating example tensors
tensor_4d = np.random.rand(3, 2, 3, 2)
tensor_3d = np.random.rand(3, 2, 2)

# Performing tensor dot product
result_tensor = np.tensordot(tensor_4d, tensor_3d, axes=([3],[2]))

# result_tensor will have dimensions 3x2x3
print(result_tensor.shape)