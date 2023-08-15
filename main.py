# Data Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, r2_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("/Users/mohammed/Downloads/train.csv")
print()
print("####################################################################")
print("================= Describe the dataset attributes ==================")
print("####################################################################")

# diversity of price range
print()
print("========== Diversity Of Price Range ==========")
priceRange = df['price_range'].unique()
print(priceRange)
print('''0 - Very Expensive
1 - Bit Expensive
2 - Affordable
3 - Easily Affordable
''')
print()

# Display all Columns and Shape of Data
print()
print("========== Display all Columns and Shape of Data ==========")
print(df.columns)
print(df.shape)
print()

# data description
print()
print("========== Data Info ==========")
print(df.info)
print()
print()

print("####################################################################")
print("======================== Project Objectives ========================")
print("####################################################################")
print("The problem wants to find out some relation between features of a mobile phone(eg:- RAM,Internal Memory etc) "
      "\nand its"
      "selling price. So it needs your help to solve this problem.")
print("In this problem you do not have to predict actual price but a price range indicating how high the price is.")
print()
print()

print("####################################################################")
print("============== Data Cleaning & Transformation Methods ==============")
print("####################################################################")

# Clean Data from null values
print()
print("========== Clean Data From Null Values & Check it ==========")
df.dropna(inplace = True)
print(df.isna().sum())
print()

# Check for Duplicates
print()
print("========== Check For Duplicates ==========")
print('Number of duplicates in data = ', df.duplicated().sum())
print()

# Screen Size in Inch
# Create New Column with merging 2 other columns
df['sc_h'] = df.sc_h / 2.54  # convert to inch
df['sc_w'] = df.sc_w / 2.54  # convert to inch
# diagonal = sqrt(pow(i,2)+pow(j,2))
schPower = np.power(df.sc_h, 2)
scwPower = np.power(df.sc_w, 2)
sci = schPower + scwPower
df["ScreenSize"] = round(np.sqrt(sci), 2)

# Convert RAM from MB to GB
df['ram'] = df.ram / 1024

# Reorder Columns
print()
print("========== Reorder & Drop Unnecessary Columns ==========")
# Drop all unnecessary columns
df.drop('sc_h', axis = 1, inplace = True)
df.drop('sc_w', axis = 1, inplace = True)
newOrder = ['battery_power', 'clock_speed', 'm_dep', 'n_cores', 'talk_time', 'fc', 'pc', 'blue', 'dual_sim',
             'four_g', 'three_g', 'touch_screen', 'wifi', 'px_height', 'px_width', 'ram', 'int_memory',
             'ScreenSize', 'mobile_wt', 'price_range']
df = df[newOrder]
print('After Reordering')
print(df['ScreenSize'].head(5))
print(df['ram'].head(5))
print()
print()

print("####################################################################")
print("=================== Anomaly Detection Technique ===================")
print("####################################################################")

# Review some Boxplot to detect outliers
print()
haveOutliers = ['fc', 'px_height']
for i in haveOutliers:
    sns.boxplot(x = df[i])
    plt.title(i)
    plt.show()


def removeOutliers(column, df):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    upper_limit = Q3 + 1.5 * IQR
    lower_limit = Q1 - 1.5 * IQR
    # Check for Outliers
    outliers = df[(df[column] < lower_limit) | (df[column] > upper_limit)]
    # Removing the outliers
    newDf = df.drop(outliers.index)
    return newDf


df = removeOutliers('fc', df)
df = removeOutliers('px_height', df)

for i in haveOutliers:
    sns.boxplot(x = df[i])
    plt.title(i)
    plt.show()

print('Now ! There\'s No Outliers.🥳')
print()
print()

print("####################################################################")
print("========================= Data Exploration =========================")
print("####################################################################")

# Review Price Ranges
print()
sns.distplot(df.battery_power, kde_kws = {'shade': True}, color = 'yellow', hist = False)
plt.title('Battery Power Mean = ' + str(round(df.battery_power.mean(), 2)))
plt.show()
sns.distplot(df.ScreenSize, kde_kws = {'shade': True}, color = 'red', hist = False)
plt.title('Screen Size in Inch Mean = ' + str(round(df.ScreenSize.mean(), 2)))
plt.show()
sns.distplot(df.int_memory, kde_kws = {'shade': True}, color = 'green', hist = False)
plt.title('Internal Memory in GB Mean = ' + str(round(df.int_memory.mean(), 2)))
plt.show()
sns.distplot(df.ram, kde_kws = {'shade': True}, color = 'purple', hist = False)
plt.title('RAM In GB Mean = ' + str(round(df.ram.mean(), 2)))
plt.show()

print("========== STD and VAR ==========")
print('Battery Power STD & VAR = ' + str(round(df.battery_power.std(), 2)) + ' , ' + str(
    round(df.battery_power.var(), 2)))
print('Screen Size STD & VAR = ' + str(round(df.ScreenSize.std(), 2)) + ' , ' + str(round(df.ScreenSize.var(), 2)))
print('Internal Memory STD & VAR = ' + str(round(df.int_memory.std(), 2)) + ' , ' + str(round(df.int_memory.var(), 2)))
print('RAM STD & VAR = ' + str(round(df.ram.std(), 2)) + ' , ' + str(round(df.ram.var(), 2)))
print()
print("========== IQR ==========")
print(
    'Battery Power IQR = ' + str(round(df.battery_power.describe()['75%'] - df.battery_power.describe()['25%'], 2)))
print('Screen Size IQR = ' + str(round(df.ScreenSize.describe()['75%'] - df.ScreenSize.describe()['25%'], 2)))
print('Internal Memory IQR = ' + str(round(df.int_memory.describe()['75%'] - df.int_memory.describe()['25%'], 2)))
print('RAM IQR = ' + str(round(df.ram.describe()['75%'] - df.ram.describe()['25%'], 2)))
print()
print()

print("####################################################################")
print("======================== Data Visualization ========================")
print("####################################################################")


def countPlot(column):
    sns.countplot(x = column, data = df)
    plt.title('What is the Count of ' + str(column))
    return plt.show()


# Count Plot of Touch Screen
countPlot('wifi')

# Count Plot of Price Range
countPlot('price_range')

print()


# Function to show a mix between histogram and density plot.
def plot(column):
    plt.figure(figsize = (10, 4))
    plt.subplot(1, 2, 1)
    sns.histplot(df[column])
    plt.subplot(1, 2, 2)
    sns.distplot(df[column])
    plt.tight_layout(pad = 1.0)
    return plt.show()


# Internal Memory
plot('n_cores')

# RAM
plot('clock_speed')

# Internal Memory VS Price Range
sns.pointplot(y = 'int_memory', x = 'price_range', data = df)
plt.title("Internal Memory VS Price Range")
plt.show()

# Pie Chart
labels4g = ["4G-supported", 'Not supported']
values4g = df['four_g'].value_counts().values
fig1, ax1 = plt.subplots()
ax1.pie(values4g, labels = labels4g, autopct = '%1.1f%%', shadow = True, startangle = 90)
plt.show()

# Cat Plot kind Count
plt.figure(figsize = (10, 6))
df['fc'].hist(alpha = 0.5, color = 'blue', label = 'Front camera')
df['pc'].hist(alpha = 0.5, color = 'red', label = 'Primary camera')
plt.legend()
plt.xlabel('MegaPixels')
plt.show()

# Screen Size vs Mobile Range
sns.jointplot(x = 'ScreenSize', y = 'price_range', data = df, kind = 'hex')
plt.show()

print('Now All Types of Visualization has done.🥳')
print()
print()

print("####################################################################")
print("===================== Machine Learning Results =====================")
print("####################################################################")

print()
print()
# split data into train and test
X = df.drop(columns = ['price_range'])
Y = df.price_range

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = .2, random_state = 42)

print("==================== XGB Classifier ====================")
# XGB Classifier
xgbClass = xgb.XGBClassifier()
xgbClass.fit(x_train, y_train)
xgbPrediction = xgbClass.predict(x_test)
# Evaluation
xgbScore = accuracy_score(y_test, xgbPrediction)
xgbR2Score = r2_score(y_test, xgbPrediction)
cn = classification_report(y_test, xgbPrediction)
print()
print('R2 Score for XGB Classifier : ', xgbR2Score)
print(cn)
print()

print("==================== Decision Tree Classifier ====================")
# Decision Tree
dTree = DecisionTreeClassifier()
dTree.fit(x_train, y_train)
dTreePrediction = dTree.predict(x_test)
# Evaluation
dTreeScore = accuracy_score(y_test, dTreePrediction)
dTreeR2Score = r2_score(y_test, dTreePrediction)
af = classification_report(y_test, dTreePrediction)
print()
print('R2 Score for Decision Tree Classifier : ', dTreeR2Score)
print(af)
print()

print("====================  Random Forest Classifier ====================")
# Random Forest
rfc = RandomForestClassifier()
rfc.fit(x_train, y_train)
rfcPrediction = rfc.predict(x_test)
# Evaluation
rfcScore = accuracy_score(y_test, rfcPrediction)
rfcR2Score = r2_score(y_test, rfcPrediction)
re = classification_report(y_test, rfcPrediction)
print()
print('R2 Score for Random Forest Classifier : ', rfcR2Score)
print(re)
