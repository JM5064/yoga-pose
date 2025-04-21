import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Original confusion matrix
conf_matrix = np.array([
    [39,  1,  4,  1,  2,  2,  1,  4,  0,  0,  0,  0],
    [ 0, 37,  0,  0,  0,  1,  1,  2,  2,  0,  0,  0],
    [ 0,  0, 25,  1,  9,  0,  1,  2,  0,  0,  3,  0],
    [ 0,  0,  2, 33,  1,  0,  2,  1,  1,  0,  2,  0],
    [ 1,  1,  9,  0, 26,  0,  2,  0,  0,  0,  2,  0],
    [ 2,  0,  0,  0,  0, 40,  1,  3,  5,  2,  1,  0],
    [ 0,  1,  0,  1,  0,  0, 39,  0,  1,  0,  0,  0],
    [ 3,  2,  4,  0,  1,  1,  1, 80,  1,  1,  1,  1],
    [ 0,  1,  0,  0,  1,  2,  0,  1, 30,  0,  1,  0],
    [ 0,  1,  0,  0,  1,  2,  0,  0,  0, 22,  0,  0],
    [ 0,  1,  0,  0,  0,  0,  0,  0,  0,  1, 34,  2],
    [ 1,  1,  0,  0,  0,  0,  1,  2,  0,  0,  3, 16],
])

# Normalize by row
row_sums = conf_matrix.sum(axis=1, keepdims=True)
conf_matrix_norm = conf_matrix / row_sums

# Class labels
labels = ["Boat Pose", "Chair Pose", "Child Pose", "Downward Facing Dog", "Fish Pose", "Lord of the Dance Pose", "Side Plank Pose", "Sitting Pose", "Tree Pose", "Warrior 3", "Warrior 2", "Warrior 1"]

# Plot
plt.figure(figsize=(10, 8))
sns.heatmap(
    conf_matrix_norm, 
    annot=True, 
    fmt=".2f", 
    cmap="Blues",
    xticklabels=labels, 
    yticklabels=labels,
    annot_kws={"size": 8}
)
plt.xlabel("Predicted Label", fontsize=10)
plt.ylabel("True Label", fontsize=10)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.title("Normalized Confusion Matrix", fontsize=12)
plt.tight_layout()
plt.show()
