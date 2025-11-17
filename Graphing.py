import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data
df = pd.read_csv('QB Data.csv')

# Create the scatter plot
plt.figure(figsize=(30, 18))
plt.scatter(df['QBR_over_Pred'], df['RTG_over_Pred'], color='blue', s=70)

# Draw midlines for quadrants
plt.axvline(x=0, color='red', linestyle='--', linewidth=1)
plt.axhline(y=0, color='red', linestyle='--', linewidth=1)

for i, row in df.iterrows():
    plt.text(row['QBR_over_Pred'], row['RTG_over_Pred'] - 0.2,  # shift label downward
             f"{row['Year']} {row['Name']}",
             fontsize=7, ha='center', va='top')  # smaller font



# Set labels
plt.xlabel('QBR over Prediction')
plt.ylabel('Passer Rating over Prediction')
plt.title('QB Performance: Beat Predictions?')

# Set axis boundaries
plt.xlim(-35, 25)
plt.ylim(-40, 30)

# Set major ticks every 5
plt.xticks(np.arange(-35, 25, 5))
plt.yticks(np.arange(-40, 30, 5))

# Set minor ticks every 2.5
plt.minorticks_on()
plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(2.5))
plt.gca().yaxis.set_minor_locator(plt.MultipleLocator(2.5))

# Grid: major = solid, minor = dotted
plt.grid(which='major', color='gray', linestyle='-', linewidth=0.8)
plt.grid(which='minor', color='gray', linestyle=':', linewidth=0.5)

plt.show()
