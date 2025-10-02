# %% [markdown]
# Import the neccessary libraries

# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# %% [markdown]
# Data Loading

# %%
# We use the "train.csv" file as it contains the target variable "price_range"
# The "test.csv" file is typically used for final submission without the target
try:
    data = pd.read_csv("train.csv")
    print("Data loaded successfully!")
except FileNotFoundError:
    print("Error: 'train.csv' was not found. Please ensure the file is in same directory")
    exit()

data

# %% [markdown]
# Data Preprocessing

# %%
# Check for missing values
data_missing = data.isnull().sum()
print("Missing Values")
print(data_missing)

# %%
# Check for duplicated rows
data_duplicated = data.duplicated().sum()
print("Duplicated Rows")
print(data_duplicated)

# %% [markdown]
# Feature Engineering

# %%
# Define the feature matrix (X) and the target variable (y)
X = data.drop("price_range",axis=1)
y = data["price_range"]

# %% [markdown]
# Visualization before Training

# %%
# Visualize the correlation matrix to understand feature realtionships
# and how features correlate with the target variable ("price_range")

plt.figure(figsize=(18,15))
# Calculate the correlation matrix for all column, including the target
correlation_matrix = data.corr()

# Draw the heatmap
sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    cbar=True,
    linewidths=0.5,
    linecolor="black"
)

plt.title("Feature Correlation Heatmap (Before Training)",fontsize=20,pad=20)
plt.show()

# %% [markdown]
# Data Splitting

# %%
# Split the data into training (80%) and testing (20%) sets
# random_state ensures reprodiucibilty of the split
X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")

# %% [markdown]
# Data Scaling

# %%
scaler = StandardScaler()

# Fit the scaler ONLY on the training data to prevent data leakage
X_train_scaled = scaler.fit_transform(X_train)

# Apply the transformation to both training and test data
X_test_scaled = scaler.transform(X_test)

# %% [markdown]
# Model Training and Comparison

# %%
# Define models to be compared
models ={
    "Logistic Regression": LogisticRegression(max_iter=1000,random_state=42),
    "K-Nearest Neighbours": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100,random_state=42),
    "Support Vector Machine": SVC(random_state=42,kernel="linear")
}

accuracy_scores = {}

for name,model in models.items():
    # Train the model using the scaled training data
    # KNN,LR and SVS need scaled data. DT and RF are scaled-invariant, but
    # we use scaled data for consitency
    model.fit(X_train_scaled,y_train)

    # Make predictions on the scaled data
    y_pred = model.predict(X_test_scaled)

    # Calculate the model's aacuracy
    accuracy = accuracy_score(y_test,y_pred)
    accuracy_scores[name] = accuracy

    # Print the result
    print(f"[{name}] Accuracy: {accuracy:.4f}")

# Find the best performing model
best_model_name = max(accuracy_scores,key=accuracy_scores.get)
best_accuracy = accuracy_scores[best_model_name]

print("-----Best Model Found-----")
print(f"The best model is {best_model_name} with an accuracy of {best_accuracy:.4f}")

# %% [markdown]
# Visualization After Training

# %%
# Convert the dictionary of results into a pandas Seies for easy plotting
results = pd.Series(accuracy_scores).sort_values(ascending=False)

plt.figure(figsize=(12,7))
# Create a bar plot of the model accuracies
bars = sns.barplot(
    x=results.index,
    y=results.values,
    palette="viridis"
)

plt.ylim(0.5,1.0)
plt.title("Classification Model Performance Comparison (Accuracy)",fontsize=16)
plt.xlabel("Classifier Model",fontsize=12)
plt.ylabel("Accuracy Score",fontsize=12)
plt.xticks(rotation=45,ha="right")
plt.tight_layout()
plt.show()


