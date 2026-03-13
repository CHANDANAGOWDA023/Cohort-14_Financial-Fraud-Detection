import matplotlib.pyplot as plt

plt.hist(df['Amount'], bins=50)
plt.title("Distribution of Transaction Amounts")
plt.xlabel("Transaction Amount")
plt.ylabel("Frequency")
plt.show()