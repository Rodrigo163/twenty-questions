import numpy as np
import pandas as pd
import sys
import random as rd

kn = pd.read_csv('knowledge_base.csv')

a = set( kn['animal'] )

# arg = sys.argv[1] # cmd line argument
# print('{0} in knowledge base:\n{1}'.format(arg, arg in a))

num_to_test = 24
print('Animals to evaluate:', rd.sample(a, num_to_test))
