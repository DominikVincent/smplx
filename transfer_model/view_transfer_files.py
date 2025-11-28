import pickle
import numpy as np

# Load the data
with open('transfer_data/smplx_to_smpl.pkl', 'rb') as f:
    data = pickle.load(f, encoding='latin1')

# 1. Exact correspondences (subset of vertices)
smpl_indices = data['valid_vertices']
smplx_indices = data['mapping']

print(len(smpl_indices), "corresponding vertices found between SMPL and SMPL-X.")
print(len(smplx_indices), "corresponding vertices found between SMPL-X and SMPL.")

# Example: Vertex i in SMPL is Vertex k in SMPL-X
for i, k in zip(smpl_indices[:5], smplx_indices[:5]):
    print(f"SMPL vertex {i} corresponds to SMPL-X vertex {k}")

# create a dictonary mapping and save it as a smpl_to_smplx_vertex_map.pkl
smpl_to_smplx_vertex_map = {smpl_idx: smplx_idx for smpl_idx, smplx_idx in zip(smpl_indices, smplx_indices)}
with open('smpl_to_smplx_vertex_map.pkl', 'wb') as f:
    pickle.dump(smpl_to_smplx_vertex_map, f)

# 2. Full reconstruction (all vertices)
# If you have SMPL-X vertices (V_smplx) of shape (N, 10475, 3)
# You can get all SMPL vertices (V_smpl) of shape (N, 6890, 3)
matrix = data['matrix'] # (6890, 10475)

# check if all rows sum to 1
row_sums = matrix.sum(axis=1)
print(all(np.isclose(row_sums, 1.0)), "All rows of the deformation transfer matrix sum to 1.")
print("This amount of rows are 1:", np.sum(row_sums == 1.0))

# get index of column that is 1 for each row
indices = np.argmax(matrix, axis=1)
print("Indices of columns that are 1 for each row:", indices[:5])

print()