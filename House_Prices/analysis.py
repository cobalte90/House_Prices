import pandas as pd
import numpy as np
import sklearn.linear_model
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import seaborn as sns
import scipy.stats as sps
from sklearn.preprocessing import OrdinalEncoder
pd.set_option('display.max_columns', None)
pd.options.display.width = 0

df = pd.read_csv('train.csv')
df = df.drop(['Alley', 'MasVnrType', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'], axis=1)
df = df.drop('MasVnrArea', axis=1)
df = df.drop('Electrical', axis=1)
df = df.drop('LandContour', axis=1)
df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].mean())
x = df.loc[:,['MSSubClass', 'SalePrice']]
df = df.dropna()

ExterQual = np.array(df['ExterQual']).reshape(-1, 1)
ExterCond = np.array(df['ExterCond']).reshape(-1, 1)
BsmtQual = np.array(df['BsmtQual']).reshape(-1, 1)
BsmtCond = np.array(df['BsmtCond']).reshape(-1, 1)
BsmtExposure = np.array(df['BsmtExposure']).reshape(-1, 1)
BsmtFinType1 = np.array(df['BsmtFinType1']).reshape(-1, 1)
BsmtFinType2 = np.array(df['BsmtFinType2']).reshape(-1, 1)
HeatingQC = np.array(df['HeatingQC']).reshape(-1, 1)
CentralAir = np.array(df['CentralAir']).reshape(-1, 1)
KitchenQual = np.array(df['KitchenQual']).reshape(-1, 1)
LotShape = np.array(df['LotShape']).reshape(-1, 1)
Utilities = np.array(df['Utilities']).reshape(-1, 1)
LandSlope = np.array(df['LandSlope']).reshape(-1, 1)
Functional = np.array(df['Functional']).reshape(-1, 1)
GarageQual = np.array(df['GarageQual']).reshape(-1, 1)
GarageCond = np.array(df['GarageCond']).reshape(-1, 1)

encoder = OrdinalEncoder(categories=[['Po', 'Fa', 'TA', 'Gd', 'Ex']])
df['ExterQual'] = encoder.fit_transform(ExterQual)
df['ExterCond'] = encoder.fit_transform(ExterCond)
df['HeatingQC'] = encoder.fit_transform(HeatingQC)
df['KitchenQual'] = encoder.fit_transform(KitchenQual)


encoder = OrdinalEncoder(categories=[['NA','Po', 'Fa', 'TA', 'Gd', 'Ex']])
df['BsmtQual'] = encoder.fit_transform(BsmtQual)
df['BsmtCond'] = encoder.fit_transform(BsmtCond)

encoder = OrdinalEncoder(categories=[['NA','No', 'Mn', 'Av', 'Gd']])
df['BsmtExposure'] = encoder.fit_transform(BsmtExposure)


encoder = OrdinalEncoder(categories=[['NA', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ']])
df['BsmtFinType1'] = encoder.fit_transform(BsmtFinType1)
df['BsmtFinType2'] = encoder.fit_transform(BsmtFinType2)

encoder = OrdinalEncoder(categories=[['N', 'Y']])
df['CentralAir'] = encoder.fit_transform(CentralAir)

encoder = OrdinalEncoder(categories=[['IR3', 'IR2', 'IR1', 'Reg']])
df['LotShape'] = encoder.fit_transform(LotShape)

encoder = OrdinalEncoder(categories=[['ELO', 'NoSeWa', 'NoSewr', 'AllPub']])
df['Utilities'] = encoder.fit_transform(Utilities)

encoder = OrdinalEncoder(categories=[['Sev', 'Mod', 'Gtl']])
df['LandSlope'] = encoder.fit_transform(LandSlope)

encoder = OrdinalEncoder(categories=[['Sal', 'Sev', 'Maj2', 'Maj1', 'Mod', 'Min2', 'Min1', 'Typ']])
df['Functional'] = encoder.fit_transform(Functional)

encoder = OrdinalEncoder(categories=[['Po', 'Fa', 'TA', 'Gd', 'Ex']])
df['GarageQual'] = encoder.fit_transform(GarageQual)
df['GarageCond'] = encoder.fit_transform(GarageCond)

df = pd.get_dummies(df, dtype='int')
model = LinearRegression()
X = df.drop('SalePrice', axis=1)
y = df['SalePrice']
model.fit(X, y)
y_pred = model.predict(X)
plt.scatter(x=y, y=y_pred)
plt.xlabel('Истинные значения')
plt.ylabel('Предсказанные моделью значения')
plt.show()
