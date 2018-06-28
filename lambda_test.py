import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# this file was initially to test lambda but now is just a basic PLT example

some_list = [1,2,3,4]
some_dict = {1:np.array([[1],[2],[3],[4]])}

list_sq = list(map(lambda x: x**2,some_list))
#print(list_sq)
#print(some_dict.values())

dict_sq = min(map(lambda x:np.amax(x),(some_dict.values())))
#print(dict_sq)

opt_dict = {12:[1,5],32:[3,1],54:[4,11]}
norms = sorted(n for n in opt_dict)

opt_choice = opt_dict[norms[0]]
print(opt_choice)
some_var = opt_choice[0]
#print(some_var)

somedf = pd.DataFrame({"A": ["cat","dog","snake","cat"],
                       "B": ["black/white","brown","white","pink"],
                       "C": [0,1,0,1]})
#for index, row in somedf.iterrows():
#    if row["A"] == "cat":
##        somedf.loc[index,"C"] = 3
#        print(somedf.loc[[index],["C"]])
#        somedf["C"].replace(12,3, inplace=True)
#print(somedf["A"].unique())

fig = plt.figure(figsize=(10,10))
i=1
for a in somedf["A"].unique():
    fig.add_subplot(3,3,i)
    plt.title("Animal: {}".format(a))
    somedf["C"][(somedf.A==a)].value_counts().plot(kind='pie')
    plt.show()
    i+=1
#    plt.plot(somedf["A"][somedf["C"]==0].count(a),(somedf["A"][somedf.C==1].count(a)))
#    plt.show()
