import pandas as pd
import matplotlib.pyplot as plt

# dummy dataset
df = pd.DataFrame({"gender":["M","F","M","M","F","M"]})

df["gender"].value_counts().plot(kind="bar")
plt.title("Gender Bias")
plt.savefig("bias.png")
plt.show()
