import numpy as np
import pandas as pd

#Simulate regression Y = 2 + X + 2*X^2
n = 50
step = 5/50
X = np.arange(-2.5, 2.5, step)
Y = 2 + 2*X*X + np.random.normal(0,1, size = n)

df_labels = pd.DataFrame(np.array([Y]).T)
df_data = pd.DataFrame(np.array([np.ones(n), X, X*X]).T)

df_labels.to_csv('./src/datasets/regression_labels.csv', index=False, header=False)
df_data.to_csv('./src/datasets/regression_data.csv', index=False, header=False)