import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
# Set Matplotlib defaults
plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)
plt.rc('animation', html='html5')

# Any results you write to the current directory are saved as output.
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

y = train_data["Survived"]

# Age and Embarked have missing values, meaning cannot use random forest
# For Age, average based on rest of categories?
features = ["Pclass", "Sex","Age", "SibSp", "Parch","Embarked"]
X = train_data[features]
X.dropna(subset='Embarked') # Only 2 records, so should be fine

# Convert to numeric categories
# Original used "get_dummies"
cat_columns = X.select_dtypes(['object']).columns
X.loc[:,cat_columns] = X[cat_columns].apply(lambda x: pd.factorize(x)[0])

toAppend = ""
con = []
for index, row in X.iterrows():
    for i in X:
        if i !="Age":
            toAppend += str(row.loc[i])
            toAppend += "-"
    con.append(toAppend)
    toAppend = ""
X.loc[:,'Con'] = con

# Set NaN ages as the average age based on all other categories
# If NaN (no matches for all categories), set as overall average for gender
ageNan = X.loc[pd.isnull(X['Age']), :]
ageNotNan = X.loc[pd.notnull(X['Age']), :]
for index, row in ageNan.iterrows():
    test = ageNotNan.loc[ageNotNan['Con']==row['Con'],'Age'].empty
    if test:
        X.loc[index,'Age'] = ageNotNan.loc[ageNotNan['Sex']==row['Sex'],'Age'].mean()
    else: 
        X.loc[index,'Age'] = ageNotNan.loc[ageNotNan['Con']==row['Con'],'Age'].mean()
X.drop('Con',axis=1,inplace=True)


# Repeat for test data
X_test = test_data[features]
cat_columns_test = X_test.select_dtypes(['object']).columns
X_test.loc[:,cat_columns] = X_test[cat_columns].apply(lambda x: pd.factorize(x)[0])

toAppend = ""
con = []
for index, row in X_test.iterrows():
    for i in X_test:
        if i !="Age":
            toAppend += str(row.loc[i])
            toAppend += "-"
    con.append(toAppend)
    toAppend = ""
    
print(X.loc[pd.isnull(X['Age']), :])
X_test.loc[:,'Con'] = con


# Set NaN ages as the average age based on all other categories
ageNan_test = X_test.loc[pd.isnull(X_test['Age']), :]
ageNotNan_test = X_test.loc[pd.notnull(X_test['Age']), :]
for index, row in ageNan_test.iterrows():
    test = ageNotNan_test.loc[ageNotNan_test['Con']==row['Con'],'Age'].empty
    if test:
        X_test.loc[index,'Age'] = ageNotNan_test.loc[ageNotNan_test['Sex']==row['Sex'],'Age'].mean()
    else: 
        X_test.loc[index,'Age'] = ageNotNan_test.loc[ageNotNan_test['Con']==row['Con'],'Age'].mean()
X_test.drop('Con',axis=1,inplace=True)


X_train, X_valid, y_train, y_valid = train_test_split(X, y, stratify=y, train_size=0.75)
input_shape = [X_train.shape[1]]

X_train = np.asarray(X_train).astype('float32')
X_valid = np.asarray(X_valid).astype('float32')
y_train = np.asarray(y_train).astype('float32')
y_valid = np.asarray(y_valid).astype('float32')
X_test = np.asarray(X_test).astype('float32')


model = keras.Sequential([
    layers.BatchNormalization(input_shape=input_shape),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(rate=0.3),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(rate=0.3), 
    layers.Dense(1, activation='sigmoid'),
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['binary_accuracy'],
)

early_stopping = keras.callbacks.EarlyStopping(
patience=5,
min_delta=0.001,
restore_best_weights=True,
) 

history = model.fit( X_train, y_train, validation_data=(X_valid, y_valid), batch_size=512, epochs=200, callbacks=[early_stopping])
history_df = pd.DataFrame(history.history) 
history_df.loc[:, ['loss', 'val_loss']].plot(title="Cross-entropy") 
history_df.loc[:, ['binary_accuracy', 'val_binary_accuracy']].plot(title="Accuracy")


predictNP = model.predict(X_test)
print(predictNP)


predictions = []
for i in predictNP:
    predictions.append(round(i[0]))

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('Intro to Deep Learning Approach.csv', index=False)
print("Your submission was successfully saved!")
