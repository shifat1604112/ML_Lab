import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import association_rules


data = pd.read_csv('Depression analysis.csv')

data_new = data.iloc[:,1:].copy()


info = data_new.to_numpy().tolist()
for values in info:
    for i in range(len(values)):
        s = ''
        if type(values[i]) != str:
            s = str(values[i])
            values[i] = s

tr = TransactionEncoder()
t_info = tr.fit(info).transform(info)
df = pd.DataFrame(t_info,columns=tr.columns_)

frequent_items = apriori(df, min_support=0.60, use_colnames=True)
result = association_rules(frequent_items, metric='confidence',
min_threshold=.60)
new_result = result.loc[:, ['antecedents', 'consequents',
'support','confidence']]
new_result.iloc[:20,:]
