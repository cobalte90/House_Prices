# House_Prices
# Подключаем необходимые библиотеки
```
import pandas as pd
import numpy as np
import sklearn.linear_model
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix
import seaborn as sns
import scipy.stats as sps
from sklearn.preprocessing import OrdinalEncoder
``` 
# Загружаем датасет, проводим обзорный анализ
```
df = pd.read_csv('train.csv')
print(df.info())
```
Видим, что некоторые колонки содержат очень мало ненулевых эелементов. Эти колонки можно удалить.
```
df = df.drop(['Alley', 'MasVnrType', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'], axis=1)
```
Также следует удалить столбец 'MasVnrArea', так как большинство значений в нём нулевые.  
В колонке 'LotFrontage' достаточно много элементов, чтобы оставить её. Пропущенные значения заменим на средние по колонке.  
Колонки 'Electrical', 'LandContour' неясно как интерпретировать, поэтому лучше их тоже удалить.
Удалим оставшиеся строки с пропущенными значениями.
```
df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].mean())
df = df.drop('MasVnrArea', axis=1)
df = df.drop('Electrical', axis=1)
df = df.drop('LandContour', axis=1)
df = df.dropna()
```
# Займёмся категориальными переменными.
## Некоторые из них, судя по описанию, являются порядковыми. Такие переменные можно закодировать с помощью OrdinalEncoder.
```
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
```
## Кодируем остальные категориальные переменные с помощью one-hot encoding.
```
df = pd.get_dummies(df, dtype='int')
```
## Создаём модель линейной регрессии, обучаем её.
```
model = LinearRegression()
X = df.drop('SalePrice', axis=1)
y = df['SalePrice']
model.fit(X, y)
```
## Предсказываем цену на дом на этих же данных, сравниваем предсказанные данные с реальными с помощью графика.
```
y_pred = model.predict(X)
plt.scatter(x=y, y=y_pred)
plt.xlabel('Истинные значения')
plt.ylabel('Предсказанные моделью значения')
plt.show()
```
![Figure_1](https://github.com/user-attachments/assets/1c98ecff-26a9-472e-9ba6-6f1e79485ccb)
## В идеале график должен принять форму прямой линии. Данная модель не является идеальной из-за большого количества переменных, но суть данного проекта - показать этапы предобработки данных и работы с категориальными переменными.
# Спасибо за внимание! Надеюсь, вам помогут какие-то способы предобработки из моего проекта.

