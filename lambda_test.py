import numpy as np

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


#print(something)
