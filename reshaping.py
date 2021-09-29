import pandas as pd
import numpy as np

df=pd.read_csv("/home/cristiano/polymer-dim-thesis/sim.csv")
n = 1000
a=np.array_split(df, n)
l = [pd.DataFrame() for x in range(n)]

for i in range(n):
    l[i] = a[i].to_numpy().flatten()


tup = (l[0],)
for i in range(1,n-1): 
    tup = tup + (l[i],)
    
#problemino perchè probabilmente legge una riga in meno
#nel csv grosso, quindi toglie 3 valori

new_df = pd.DataFrame(np.vstack(tup))
new_df.to_csv("ultimate.csv", index=False)