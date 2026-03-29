import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# 1. Put your data into a Pandas DataFrame format
data = pd.DataFrame(
    {
        "Model": ["Baseline"] * 8 + ["New Technique"] * 8,
        "AUC": [0.85, 0.86, 0.84, 0.87, 0.86, 0.85, 0.88, 0.84]
        + [0.89, 0.91, 0.88, 0.90, 0.89, 0.92, 0.90, 0.88],
    }
)

plt.figure(figsize=(4, 6))

# 2. Draw the box plot
sns.boxplot(x="Model", y="AUC", data=data, palette="Set2", width=0.3)

# 3. Overlay the actual 8 individual scores as dots
sns.stripplot(
    x="Model", y="AUC", data=data, color="black", alpha=0.6, jitter=True, size=6
)

plt.ylabel("AUC Score", fontsize=12)
plt.xlabel("", fontsize=12)
plt.title("Model Performance Comparison", fontsize=14)

# plt.savefig("advanced_boxplot.png")
plt.show()
