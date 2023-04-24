import pandas as pd
import numpy as np
from io import StringIO
from sklearn.impute import SimpleImputer

csv_data = '''A,B,C,D
              1.0,2.0,3.0,4.0
              5.0,6.0,,8.0
              10.0,11.0,12.0,'''

df = pd.read_csv(StringIO(csv_data))

# SimpleImputer のインスタンスを作成
imr = SimpleImputer(missing_values=np.nan, strategy='mean')
imr = imr.fit(df.values)
imputed_data = imr.transform(df.values)

df2 = pd.DataFrame([['green', 'M', 10.1, 'class1'],
                    ['red', 'L', 13.5, 'class2'],
                    ['blue', 'XL', 15.3, 'class2']])
df2.columns = ['color', 'size', 'price', 'classlabel']

size_mapping = {'XL':3, 'L':2, 'M':1}
df2['size'] = df2['size'].map(size_mapping)

class_mapping = {label:idx for idx, label in enumerate(np.unique(df2['classlabel']))}
df2['classlabel'] = df2['classlabel'].map(class_mapping)

X = df2[['color', 'size', 'price']].values
df2 = pd.get_dummies(df2[['price', 'color', 'size']], drop_first=True)
print(df2)
