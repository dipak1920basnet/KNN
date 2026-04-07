# %% ================================
# 1. Imports
# ==================================
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# %% ================================
# 2. Create Dataset
# ==================================
X, y = make_moons(n_samples=300, noise=0.3, random_state=42)

df = pd.DataFrame(X, columns=["Feature 1", "Feature 2"])
df["Target"] = y

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x="Feature 1", y="Feature 2", hue="Target", palette="Set1")
plt.title("2D Classification Data (make_moons)")
plt.grid(True)
plt.show()

# %% ================================
# 3. Train-Test Split (FIRST)
# ==================================
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# %% ================================
# 4. Scaling (FIT ONLY ON TRAIN)
# ==================================
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)   # learn from train
X_test_scaled = scaler.transform(X_test)         # apply to test

# %% ================================
# 5. Find Best k using Cross-Validation (ONLY TRAIN DATA)
# ==================================
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

k_range = range(1, 21)
cv_scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    
    scores = cross_val_score(
        knn,
        X_train_scaled,   # ONLY TRAIN DATA
        y_train,
        cv=5,
        scoring='accuracy'
    )
    
    cv_scores.append(scores.mean())

# Plot k vs accuracy
plt.figure(figsize=(8, 5))
plt.plot(k_range, cv_scores, marker='o')
plt.title("k-NN Cross-Validation Accuracy vs k")
plt.xlabel("Number of Neighbors (k)")
plt.ylabel("Accuracy")
plt.grid(True)
plt.show()

# Best k
best_k = k_range[np.argmax(cv_scores)]
print(f"Best k: {best_k}")

# %% ================================
# 6. Train Final Model
# ==================================
best_knn = KNeighborsClassifier(n_neighbors=best_k)
best_knn.fit(X_train_scaled, y_train)

# %% ================================
# 7. Evaluate on Test Data
# ==================================
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay

y_pred = best_knn.predict(X_test_scaled)

print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.2f}")

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=["Class 0", "Class 1"])
disp.plot(cmap="Blues")
plt.title(f"Confusion Matrix (k={best_k})")
plt.grid(False)
plt.show()

print("Classification Report:")
print(classification_report(y_test, y_pred))

# %% ================================
# 8. Decision Boundary (IMPORTANT)
# ==================================

# Create mesh grid (IN ORIGINAL SPACE)
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 300),
    np.linspace(y_min, y_max, 300)
)

# IMPORTANT: scale grid before prediction
grid_points = np.c_[xx.ravel(), yy.ravel()]
grid_points_scaled = scaler.transform(grid_points)

Z = best_knn.predict(grid_points_scaled)
Z = Z.reshape(xx.shape)

# Plot
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.3)
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette="Set1", edgecolor='k')

plt.title(f"Decision Boundary (k={best_k})")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.grid(True)
plt.show()