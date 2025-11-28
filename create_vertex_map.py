import pickle
import numpy as np

# Load the data
file_path = 'transfer_data/smplx_to_smpl.pkl'
with open(file_path, 'rb') as f:
    data = pickle.load(f, encoding='latin1')

# Extract arrays
smpl_indices = data['valid_vertices']
smplx_indices = data['mapping']
matrix = data['matrix']

print(f"Total SMPL vertices: {matrix.shape[0]}")
print(f"Exact matches found: {len(smpl_indices)}")

# Strategy:
# 1. For exact matches, use the provided mapping (which corresponds to matrix value ~1.0)
# 2. For non-exact matches, find the SMPL-X vertex with the highest weight in the transfer matrix
#    This corresponds to the "closest" or most influential vertex.

# Get the argmax for every row in the matrix
# This gives the index of the SMPL-X vertex with the largest weight for each SMPL vertex
all_closest_indices = np.argmax(matrix, axis=1)

# Verify that for the exact matches, the argmax gives the same result
# Note: matrix is (6890, 10475). all_closest_indices is (6890,)
matches_check = all_closest_indices[smpl_indices]
mismatch_count = np.sum(matches_check != smplx_indices)
print(f"Mismatch between argmax and provided mapping for exact matches: {mismatch_count}")

if mismatch_count > 0:
    print("Warning: argmax does not perfectly align with exact mapping. Preferring exact mapping where available.")

# Create the full map
# Initialize with the argmax for everything (covers the non-exact ones)
full_map = {i: int(idx) for i, idx in enumerate(all_closest_indices)}

# Overwrite with the exact matches to be safe (though they should be identical/better)
for smpl_idx, smplx_idx in zip(smpl_indices, smplx_indices):
    full_map[smpl_idx] = int(smplx_idx)

# Save dictionary
output_path = 'smpl_to_smplx_vertex_map.pkl'
with open(output_path, 'wb') as f:
    pickle.dump(full_map, f)

print(f"Saved mapping dictionary to {output_path}")
print(f"Dictionary size: {len(full_map)}")
print(f"Example entry for missing vertex (e.g. 0): {full_map[0]}")

