import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

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


# Split into validation and training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# Specify Model
model = DecisionTreeRegressor(random_state=1)
# Fit Model
model.fit(train_X, train_y)

val_predictions = model.predict(val_X)
val_mae = mean_absolute_error(val_predictions, val_y)
print("Validation MAE when not specifying max_leaf_nodes: {:,.0f}".format(val_mae))

# Using best value for max_leaf_nodes
model = DecisionTreeRegressor(max_leaf_nodes=100, random_state=1)
model.fit(train_X, train_y)
val_predictions = model.predict(val_X)
val_mae = mean_absolute_error(val_predictions, val_y)
print("Validation MAE for best value of max_leaf_nodes: {:,.0f}".format(val_mae))

from sklearn.ensemble import RandomForestRegressor

# Define the model. Set random_state to 1
rf_model = RandomForestRegressor(random_state=1)

# fit your model
rf_model.fit(train_X, train_y)

# Calculate the mean absolute error of your Random Forest model on the validation data
rf_preds = rf_model.predict(val_X)
rf_val_mae = mean_absolute_error(val_y, rf_preds)

print("Validation MAE for Random Forest Model: {}".format(rf_val_mae))

predictions = model.predict(X_test)
predictions = predictions.round(0)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('Intro to Machine Learning Approach.csv', index=False)
print("Your submission was successfully saved!")
