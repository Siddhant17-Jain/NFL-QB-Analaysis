import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data
df = pd.read_csv('QB_25.csv')

# Create the scatter plot
plt.figure(figsize=(30, 18))
plt.scatter(df['Predicted_QBR'], df['Predicted_RTG'], color='blue', s=70)

# Draw midlines for quadrants
plt.axvline(x=57.5, color='red', linestyle='--', linewidth=1)
plt.axhline(y=95, color='red', linestyle='--', linewidth=1)

for i, row in df.iterrows():
    plt.text(row['Predicted_QBR'], row['Predicted_RTG'] - 0.2,  # shift label downward
             f"{row['Name']}",
             fontsize=7, ha='center', va='top')  # smaller font



# Set labels
plt.xlabel('Predicted_QBR')
plt.ylabel('Predicted_RTG')
plt.title('QB Performance: Best Situations')

# Set axis boundaries
plt.xlim(45, 70)
plt.ylim(85, 105)

# Set major ticks every 5
plt.xticks(np.arange(45, 70, 5))
plt.yticks(np.arange(85, 105, 5))

# Set minor ticks every 2.5
plt.minorticks_on()
plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(2.5))
plt.gca().yaxis.set_minor_locator(plt.MultipleLocator(2.5))

# Grid: major = solid, minor = dotted
plt.grid(which='major', color='gray', linestyle='-', linewidth=0.8)
plt.grid(which='minor', color='gray', linestyle=':', linewidth=0.5)

plt.show()
