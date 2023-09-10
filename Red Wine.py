#!/usr/bin/env python
# coding: utf-8

# # Red Wine Project

# **Project Description:** The dataset is related to red and white variants of the Portuguese "Vinho Verde" wine. Due to privacy and logistic issues, only physicochemical (inputs) and sensory (the output) variables are available (e.g. there is no data about grape types, wine brand, wine selling price, etc.).
# 
# This dataset can be viewed as classification task. The classes are ordered and not balanced (e.g. there are many more normal wines than excellent or poor ones). Also, we are not sure if all input variables are relevant. So it could be interesting to test feature selection methods.

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# In[2]:


df=pd.read_csv("https://raw.githubusercontent.com/dsrscientist/DSData/master/winequality-red.csv")
df


# In[3]:


df.head


# Observation 
# > The dataset contains 12 columns and 1599 rows

# In[4]:


df.shape


# In[5]:


df.dtypes


# Observation 
# > This dataset contains 12 columns contains numerical datatype."quality" is the target variable and dependent column.and  Others  independent columns.
# > Only "quality" variable id integar datatype and  all the columns are float datatype.

# In[79]:


df.columns.to_list()


# In[7]:


df.info()


# # Missing Values

# df.isnull().sum()

# In[9]:


sns.heatmap(df.isnull())
plt.show()


# Observations 
# > No null values in the dataset.
# > All the columns has numerical datatype.
# 

# # Duplicate Values

# In[75]:


# Check duplicate values in the dataset
print("its showing {} duplicates values present in the dataset".format(df.duplicated().sum()))


# In[76]:


#Dropping duplicated Values
df.drop_duplicates(inplace=True)


# In[77]:


#check Duplicated Values again
df.duplicated().sum()


# In[78]:


df.shape


# Observations 
# 
# > We show 240 duplicated values present in the dataset and we have removed all the duplicate values in the dataframe.
# > After removing the duplicated values. 
# > We have 1359 rows and 12 columns.

# ## Descriptive Statistics

# In[14]:


df.describe()


# Observations
# > Target variable/Dependent variable is discrete and  also categorical in nature.
# > Mean is more than median(50% percentile) in all the columns that means data is positively skewed.
# > Huge difference in 75% percentile and max value in 'residual sugar','free sulfar dioxide' and 'total sulfur dioxide' this indicates outliers present.

# In[15]:


# check unique values
df.nunique()


#  Observations
# 
# - **Fixed Acidity:** This column appears to have a moderate number of unique values (96). its likely a continuous variable with a relatively wide range of variation.
# - **Volatile Acidity:** The large number of unique values (143) suggests a wider range of variation compared to fixed acidity.This might also be a continuous variable with a broader distribution.
# - **Citric Acid:** It has  80 unique values, citric acid seems to have a relatively smaller range of variation compared to volatile acidity. It could be a continuous variable.
# - **Residual Sugar:**It's column has 91 unique values,indicating a moderate range of variation. Its likely a continuous variable, possibly representing the amount of residual sugar in the wine.
# - **Chlorides:** High number of unique values (153) suggests a wide range of variation in chloride content.it has probably a continuous variable.
# - **Free Sulfur Dioxide:** it has 60 unique values,Column has a relatively smaller range of variation. It might represent the amount of free sulfur dioxide in the wine and could be a continuous variable.
# - **Total Sulfur Dioxide:** It has 144 unique values indicate a moderate range of variation. This column might represent the total sulfur dioxide content and could be a continuous variable.
# - **Density:** It has 436 unique values indicate a wide range of variation in density. It's likely a continuous variable, and the wide range might indicate some outliers.
# - **pH:** It has 89 unique values, pH also appears to have a relatively smaller range of variation. It represents the acidity or alkalinity of the wine and is probably a continuous variable.
# - **Sulphates:** Column has 96 unique values, similar to fixed acidity. It would be another continuous variable.
# - **Alcohol:** It has 65 unique values, alcohol content seems to have a moderate range of variation. It's probably a continuous variable representing the alcohol percentage.
# - **Quality:** Column has 6 unique values, indicating it's a discrete variable representing different quality ratings. It might be used as the target variable for classification tasks or as a dependent variable in regression tasks.

# #### Bivariate Analysis:

# In[16]:


# value_count of quality Traget Variable
print('Unique values present in alcohol :', len(df['quality'].value_counts()))
print(df['quality'].value_counts())
sns.countplot(x='quality', data=df)
plt.show()


# Observations
# > The dataset exhibits an imbalance in the distribution of quality ratings.Majority of observations fall into the middle-quality ratings of 5 and 6, with significantly fewer occurrences in the lower (3, 4) and higher (7, 8) quality ratings. This imbalance is important to consider when performing any classification or modeling tasks involving quality as the target variable.

# In[17]:


#Exploring Alcohol variable
print('Unique values present in alcohol :',len(df['alcohol'].unique()))
print(df['alcohol'].value_counts(bins=5))
df['alcohol'].value_counts(bins=5).plot(kind='pie',autopct='%0.1f%%')
plt.show()


# Observations
# - The "Alcohol" column 65 unique values, Indicating a moderate range of variation in alcohol content across the dataset.
# - The data is divided into five intervals (bins) based on alcohol content ranges:
#     1. (9.7, 11.0] Interval: There are 528 records falling within this range of alcohol content, suggesting that this range is the most common in the dataset.
#     2. (8.393, 9.7] Interval: There are 473 records falling within this lower range of alcohol content.
#     3. (11.0, 12.3] Interval: There are 265 records falling within this mid-range of alcohol content.
#     4. (12.3, 13.6] Interval: There are 86 records falling within this higher mid-range of alcohol content.
#     5. (13.6, 14.9] Interval: There are only 7 records with a relatively high alcohol content within this range.
# - The frequency distribution shows that the majority of observations fall into the two middle intervals, with decreasing frequency as we move towards the lower and higher alcohol content ranges. This suggests that wines in the mid-range of alcohol content are more prevalent in the dataset, while wines with extremely low or high alcohol content are less common.

# In[18]:


#Exploring 'Sulphates' variable
print("Unique values present in Sulphate",len(df['sulphates'].unique()))
print(df['sulphates'].value_counts(bins=5))
df['sulphates'].value_counts(bins=5).plot(kind='pie',autopct='%0.1f%%')
plt.show()


# Observations 
# - The "Sulphate" column has 96 unique values, indicating a wide range of variation in sulphate content across the dataset.
# - Frequency Distribution: The data is divided into five intervals (bins) based on sulphate content ranges, and the distribution is further analyzed in relation to wine quality:
#     1. (0.327, 0.664] Interval: In this range of sulphate content, there are 858 records. This indicates that the majority of wines with sulphate content in this interval are present in the dataset.
#     2. (0.664, 0.998] Interval: There are 446 records within this range, suggesting a significant presence of wines with slightly higher sulphate content.
#     3. (0.998, 1.332] Interval: Within this sulphate content range, there are 44 records. This range represents wines with a moderate sulphate content.
#     4. (1.332, 1.666] Interval: There are only 8 records within this interval, indicating that wines with sulphate content in this range are relatively rare in the dataset.
#     5. (1.666, 2.0] Interval: This range has the fewest records, with only 3 wines falling into this category, representing wines with a relatively high sulphate content.
# - The distribution of sulphate content ranges in relation to wine quality is not explicitly mentioned in the provided information. However, we can further analyze this data to see if there are any patterns or correlations between sulphate content and wine quality. It's possible that certain sulphate content ranges are more common in wines of particular quality ratings.

# In[19]:


#Exploring pH variable
print("Unique values present in pH",len(df['pH'].unique()))
print(df['pH'].value_counts(bins=5))
bins = pd.cut(df['pH'], bins=5)
sns.countplot(x=bins, data=df)
plt.show()


# Observations
# - The "pH" column has 89 unique values, indicating a range of variation in pH levels across the dataset.
# - Frequency Distribution and pH Intervals: The pH data is divided into five intervals (bins), and the distribution within each interval is provided:
#     1. (3.248, 3.502] Interval: This pH range is the most common, with 764 records. It suggests that a significant portion of the wines in the dataset falls within this pH range.
#     2. (2.994, 3.248] Interval: The second most common pH range contains 428 records, indicating a substantial presence of wines with slightly lower pH levels.
#     3. (3.502, 3.756] Interval: There are 136 records in this pH range, suggesting wines with slightly higher pH levels, but they are less common than those in the first two ranges.
#     4. (2.738, 2.994] Interval: This interval contains 24 records, representing wines with relatively lower pH levels.
#     5. (3.756, 4.01] Interval: The pH levels in this range are the least common, with only 7 records in this category, indicating that wines with higher pH levels are relatively rare in the dataset.
# - The provided information focuses on the distribution of pH levels but does not explicitly mention the relationship between pH levels and wine quality. However, we can further analyze this data to explore if certain pH level ranges are associated with specific wine quality ratings. For instance, do wines with higher or lower pH levels tend to receive higher or lower quality ratings?

# In[20]:


#Eexploring Density variable
print("Unique Values in density variable is :", len(df['density'].value_counts()))
print(df['density'].value_counts(bins=5))
plt.hist(df['density'], bins=5, edgecolor='k')
plt.show()


# Observations
# - Unique Values in Density: The "density" column has a total of 436 unique values, indicating a wide range of variation in density across the dataset.
# - The data is divided into five intervals (bins) based on density values, and the distribution within each interval is provided:
#     1. (0.996, 0.998] Interval: This density range contains 800 records, making it the most common density interval in the dataset. Wines within this density range are prevalent.
#     2. (0.993, 0.996] Interval: There are 292 records within this density range, indicating a substantial presence of wines with slightly lower density values compared to the most common range.
#     3. (0.998, 1.001] Interval: This interval contains 213 records, suggesting wines with slightly higher density values than the most common range.
#     4. (0.989, 0.993] Interval: With only 31 records in this range, wines with lower density values are relatively rare in the dataset.
#     5. (1.001, 1.004] Interval: The density levels in this range are the least common, with only 23 records in this category, indicating that wines with higher density values are relatively rare.
# - We can investigate if there are any patterns or correlations between density and wine quality. For example, we can explore if wines with certain density ranges tend to have higher or lower quality ratings.

# In[21]:


#Exploring Total sulfur dioxide variable
print("Unique Values in total sulfur dioxide variable is :", len(df['total sulfur dioxide'].value_counts()))
print(df['total sulfur dioxide'].value_counts(bins=5))
plt.hist(df['total sulfur dioxide'], bins=5, edgecolor='k')
plt.show()


# Observations
# - The "total sulfur dioxide" column has 144 unique values, indicating a moderate range of variation in total sulfur dioxide levels across the dataset.
# - The data is divided into five intervals (bins) based on total sulfur dioxide values, and the distribution within each interval is provided:
#     1. (5.716, 62.6] Interval: This interval contains 1015 records, making it the most common range for total sulfur dioxide levels. It suggests that a significant portion of wines in the dataset falls within this range.
#     2. (62.6, 119.2] Interval: There are 288 records within this range, indicating a presence of wines with higher total sulfur dioxide levels compared to the most common range, although they are less common.
#     3. (119.2, 175.8] Interval: This interval contains 54 records, suggesting wines with higher total sulfur dioxide levels but representing a relatively small portion of the dataset.
#     4. (232.4, 289.0] Interval: With only 2 records in this range, wines with total sulfur dioxide levels in this high range are very rare in the dataset.
#     5. (175.8, 232.4] Interval: Interestingly, this interval has zero records, indicating that there are no wines falling into this particular range in the dataset.
# - We can explore if there are any patterns or correlations between total sulfur dioxide levels and wine quality. For instance, we can investigate if wines with higher or lower total sulfur dioxide levels tend to have higher or lower quality ratings.

# In[22]:


#exploring 'free sulfur dioxide' variable
print("Unique Values in free sulfur dioxide variable is :", len(df['free sulfur dioxide'].value_counts()))
print(df['free sulfur dioxide'].value_counts(bins=5))
plt.hist(df['free sulfur dioxide'], bins=5, edgecolor='k')
plt.show()


# **Observations :** 
# * The "free sulfur dioxide" column has 60 unique values, indicating a relatively smaller range of variation in free sulfur dioxide levels across the dataset.
# * The data is divided into five intervals (bins) based on free sulfur dioxide values, and the distribution within each interval is provided:
#     1. (0.928, 15.2] Interval: This interval contains 781 records, making it the most common range for free sulfur dioxide levels. It suggests that a significant portion of wines in the dataset falls within this range.
#     2. (15.2, 29.4] Interval: There are 426 records within this range, indicating a presence of wines with moderately higher free sulfur dioxide levels compared to the most common range, although they are less common.
#     3. (29.4, 43.6] Interval: This interval contains 129 records, suggesting wines with higher free sulfur dioxide levels but representing a smaller portion of the dataset.
#     4. (43.6, 57.8] Interval: With only 20 records in this range, wines with free sulfur dioxide levels in this higher range are relatively rare in the dataset.
#     5. (57.8, 72.0] Interval: This interval has the fewest records, with only 3 wines falling into this category, indicating that wines with the highest free sulfur dioxide levels are extremely rare.
# * We can explore if there are any patterns or correlations between free sulfur dioxide levels and wine quality. For example, you can investigate if wines with higher or lower free sulfur dioxide levels tend to have higher or lower quality ratings.

# In[23]:


#Exploring chlorides variable
print("Unique Values in chlorides variable is :", len(df['chlorides'].value_counts()))
print(df['chlorides'].value_counts(bins=5))
plt.hist(df['chlorides'], bins=5, edgecolor='k')
plt.show()


# Observations
# - The "chlorides" column has 153 unique values, indicating a wide range of variation in chloride levels across the dataset.
# - Frequency Distribution: The data is divided into five intervals (bins) based on chloride values, and the distribution within each interval is provided:
#     1. (0.0104, 0.132] Interval: This interval contains 1286 records, making it the most common range for chloride levels. It suggests that a significant portion of wines in the dataset falls within this range.
#     2. (0.132, 0.252] Interval: There are 49 records within this range, indicating a presence of wines with moderately higher chloride levels compared to the most common range, although they are less common.
#     3. (0.252, 0.371] Interval: This interval contains 11 records, suggesting wines with higher chloride levels but representing a relatively small portion of the dataset.
#     4. (0.371, 0.491] Interval: With 11 records in this range, wines with chloride levels in this higher range are relatively rare in the dataset.
#     5. (0.491, 0.611] Interval: This interval has the fewest records, with only 2 wines falling into this category, indicating that wines with the highest chloride levels are extremely rare.
# - We can explore if there are any patterns or correlations between chloride levels and wine quality. For example, we can investigate if wines with higher or lower chloride levels tend to have higher or lower quality ratings.

# In[24]:


#Exploring Residual sugars variable
print("Unique Values in residual sugar variable is :", len(df['residual sugar'].value_counts()))
print(df['residual sugar'].value_counts(bins=5))
plt.hist(df['residual sugar'], bins=5, edgecolor='k')
plt.show()


# Observations
# - The "Residual sugar" column has 91 unique values, indicating a moderate range of variation in residual sugar levels across the dataset.
# - The data is divided into five intervals (bins) based on residual sugar values, and the distribution within each interval is provided:
#     1. (0.884, 3.82] Interval: This interval contains 1246 records, making it the most common range for residual sugar levels. It suggests that a significant portion of wines in the dataset falls within this range.
#     2. (3.82, 6.74] Interval: There are 89 records within this range, indicating a presence of wines with moderately higher residual sugar levels compared to the most common range, although they are less common.
#     3. (6.74, 9.66] Interval: This interval contains 16 records, suggesting wines with higher residual sugar levels but representing a relatively small portion of the dataset.
#     4. (12.58, 15.5] Interval: With only 6 records in this range, wines with very high residual sugar levels are rare in the dataset.
#     5. (9.66, 12.58] Interval: This interval has the fewest records, with only 2 wines falling into this category, indicating that wines with extremely high residual sugar levels are extremely rare.
# - We can explore if there are any patterns or correlations between residual sugar levels and wine quality. For example, we can investigate if wines with higher or lower residual sugar levels tend to have higher or lower quality ratings.

# In[25]:


#Exploring Citric acid variable
print("Unique Values in citric acid variable is :", len(df['citric acid'].value_counts()))
print(df['citric acid'].value_counts(bins=5))
plt.hist(df['citric acid'], bins=5, edgecolor='k')
plt.show()


# Observation
# - The "citric acid" column has 80 unique values, indicating a relatively smaller range of variation in citric acid levels across the dataset.
# - The data is divided into five intervals (bins) based on citric acid values, and the distribution within each interval is provided:
#     1. (-0.002, 0.2] Interval: This interval contains 539 records, making it the most common range for citric acid levels. It suggests that a significant portion of wines in the dataset falls within this range.
#     2. (0.2, 0.4] Interval: There are 438 records within this range, indicating a presence of wines with moderately higher citric acid levels compared to the most common range, although they are less common.
#     3. (0.4, 0.6] Interval: This interval contains 316 records, suggesting wines with higher citric acid levels but representing a relatively small portion of the dataset.
#     4. (0.6, 0.8] Interval: With only 65 records in this range, wines with even higher citric acid levels are relatively rare in the dataset.
#     5. (0.8, 1.0] Interval: This interval has the fewest records, with only 1 wine falling into this category, indicating that wines with extremely high citric acid levels are extremely rare.
# - We can explore if there are any patterns or correlations between citric acid levels and wine quality. For example, we can investigate if wines with higher or lower citric acid levels tend to have higher or lower quality ratings.

# In[26]:


#Exploring Volatile acidity variable
print("Unique Values in volatile acidity variable is :", len(df['volatile acidity'].value_counts()))
print(df['volatile acidity'].value_counts(bins=5))
plt.hist(df['volatile acidity'], bins=5, edgecolor='k')
plt.show()


# Observations
# - The "volatile acidity" column has 143 unique values, indicating a wider range of variation in volatile acidity levels across the dataset.
# - The data is divided into five intervals (bins) based on volatile acidity values, and the distribution within each interval is provided:
#     1. (0.412, 0.704] Interval: This interval contains 742 records, making it the most common range for volatile acidity levels. It suggests that a significant portion of wines in the dataset falls within this range.
#     2. (0.118, 0.412] Interval: There are 415 records within this range, indicating a presence of wines with moderately lower volatile acidity levels compared to the most common range, although they are less common.
#     3. (0.704, 0.996] Interval: This interval contains 179 records, suggesting wines with moderately higher volatile acidity levels but representing a relatively small portion of the dataset.
#     4. (0.996, 1.288] Interval: With only 20 records in this range, wines with higher volatile acidity levels are relatively rare in the dataset.
#     5. (1.288, 1.58] Interval: This interval has the fewest records, with only 3 wines falling into this category, indicating that wines with extremely high volatile acidity levels are extremely rare.
# - We can explore if there are any patterns or correlations between volatile acidity levels and wine quality. For example, We can investigate if wines with higher or lower volatile acidity levels tend to have higher or lower quality ratings.

# In[27]:


#Exploring fixed acidity variable
print("Unique Values in fixed acidity variable is :", len(df['fixed acidity'].value_counts()))
print(df['fixed acidity'].value_counts(bins=5))
plt.hist(df['fixed acidity'], bins=5, edgecolor='k')
plt.show()


# Observations
# -The "fixed acidity" column has 96 unique values, indicating a moderate range of variation in fixed acidity levels across the dataset.
# - The data is divided into five intervals (bins) based on fixed acidity values, and the distribution within each interval is provided:
#     1. (6.86, 9.12] Interval: This interval contains 771 records, making it the most common range for fixed acidity levels. It suggests that a significant portion of wines in the dataset falls within this range.
#     2. (9.12, 11.38] Interval: There are 255 records within this range, indicating a presence of wines with moderately higher fixed acidity levels compared to the most common range, although they are less common.
#     3. (4.588, 6.86] Interval: This interval contains 237 records, suggesting wines with moderately lower fixed acidity levels compared to the most common range.
#     4. (11.38, 13.64] Interval: With only 87 records in this range, wines with higher fixed acidity levels are relatively rare in the dataset.
#     5. (13.64, 15.9] Interval: This interval has the fewest records, with only 9 wines falling into this category, indicating that wines with extremely high fixed acidity levels are extremely rare.
# - We can explore if there are any patterns or correlations between fixed acidity levels and wine quality. For example, we can investigate if wines with higher or lower fixed acidity levels tend to have higher or lower quality ratings.

# #### Bivariate Analysis

# In[28]:


df.groupby('quality').mean().T


# In[29]:


#Visualization with barplot
plt.figure(figsize=(20, 15))
p=1
for i in df.columns[:-1]:
    if p <= 12:
        plt.subplot(4,3,p)
        sns.barplot(x='quality', y=i, data=df)
        plt.xticks()
        plt.title(f'barplot of {i} vs. Wine Quality')
        plt.xlabel('Wine Quality')
        plt.ylabel(i)
        p += 1
plt.tight_layout()
plt.show()


#  Observations 
# - **Fixed Acidity:** Wine quality increases from 4 to 8, there is a slight upward trend in fixed acidity levels. Wines with higher quality ratings tend to have slightly higher fixed acidity.
# - **Volatile Acidity:** It has a noticeable decrease in volatile acidity levels as wine quality improves. Wines with higher quality ratings (7 and 8) have significantly lower volatile acidity compared to wines with lower ratings (4 and 5).
# - **Citric Acid:** Citric acid levels generally increase as wine quality improves. Higher-quality wines (7 and 8) tend to have higher citric acid content compared to lower-quality wines (4 and 5).
# - **Residual Sugar:** It has no strong trend in residual sugar levels with respect to wine quality. The differences in residual sugar content among different quality ratings are relatively small.
# - **Chlorides:** Chloride levels show a slight decrease as wine quality increases. Higher-quality wines (7 and 8) tend to have slightly lower chloride content compared to lower-quality wines (4 and 5).
# - **Free Sulfur Dioxide:** It has  minor decrease in free sulfur dioxide levels as wine quality improves. Higher-quality wines have slightly lower free sulfur dioxide content.
# - **Total Sulfur Dioxide:** Total sulfur dioxide levels also show a minor decrease as wine quality increases. Wines with higher quality ratings tend to have slightly lower total sulfur dioxide levels.
# - **Density:** Density remains relatively consistent across different wine quality ratings. There is no strong trend in density with respect to wine quality.
# - **pH:** pH levels show a subtle decrease as wine quality improves. Higher-quality wines tend to have slightly lower pH values.
# - **Sulphates:** There is a trend of increasing sulphate levels as wine quality improves. Higher-quality wines (7 and 8) tend to have higher sulphate content compared to lower-quality wines (4 and 5).
# - **Alcohol:** Alcohol content significantly increases with higher wine quality ratings. Wines with higher quality ratings (7 and 8) have notably higher alcohol content compared to lower-quality wines (4 and 5).
# 
# These observations provide insights into how various wine characteristics relate to wine quality ratings. For example, lower volatile acidity and higher alcohol content are associated with higher-quality wines, while other characteristics such as fixed acidity and citric acid also exhibit some differences across quality ratings. These insights can be valuable for understanding the factors that contribute to wine quality and for making predictions or recommendations related to wine quality based on these characteristics.

# In[30]:


#Correlation between target variable & independent variables.
dfcor=df.corr()
dfcor


# In[80]:


#lets Visualize with Heatmap
plt.figure(figsize=(10,6))
sns.heatmap(dfcor,cmap="Blues",annot=True)
plt.show()


# In[ ]:


Observations
- 1.  From the above plot, it has show that pH & total sulfur dioxide are much rellated to our label i.e. quality.
- 2. Quality is positively related with alcohol, sulphates, residual sugar, citric acid and fixed acidity. 
- 3. Quality is negatively related with pH, density, total sulfur dioxide, free sulfur dioxide, chlorides & volatile acidity. 


# In[32]:


corQua= df.corr()['quality'].sort_values()
corQua


# Observations
# 
# 1. **Fixed Acidity vs. Quality:** There is a moderate positive correlation (0.12) between fixed acidity and wine quality. This suggests that wines with slightly higher fixed acidity may tend to have slightly higher quality ratings.
# 2. **Volatile Acidity vs. Quality:** There is a moderate negative correlation (-0.40) between volatile acidity and wine quality. As volatile acidity increases, wine quality tends to decrease. Wines with lower volatile acidity are associated with higher quality ratings.
# 3. **Citric Acid vs. Quality:** Citric acid shows a moderate positive correlation (0.23) with wine quality. Wines with higher citric acid content tend to have slightly higher quality ratings.
# 4. **Residual Sugar vs. Quality:** Residual sugar has a very weak positive correlation (0.01) with wine quality. There is almost no significant relationship between residual sugar and quality.
# 5. **Chlorides vs. Quality:** Chlorides exhibit a weak negative correlation (-0.13) with wine quality. Higher chloride levels are associated with slightly lower quality ratings.
# 6. **Free Sulfur Dioxide vs. Quality:** Free sulfur dioxide has a weak negative correlation (-0.05) with wine quality. The relationship is not very strong, suggesting that there's little impact of free sulfur dioxide on wine quality.
# 7. **Total Sulfur Dioxide vs. Quality:** Total sulfur dioxide shows a weak negative correlation (-0.18) with wine quality. Higher total sulfur dioxide levels are associated with slightly lower quality ratings.
# 8. **Density vs. Quality:** Density has a moderate negative correlation (-0.18) with wine quality. Lower density values are associated with higher quality wines.
# 9. **pH vs. Quality:** pH has a weak negative correlation (-0.06) with wine quality. Wines with slightly lower pH values tend to have slightly higher quality ratings.
# 10. **Sulphates vs. Quality:** Sulphates show a moderate positive correlation (0.25) with wine quality. Higher sulphate levels are associated with higher quality ratings.
# 11. **Alcohol vs. Quality:** Alcohol exhibits a strong positive correlation (0.48) with wine quality. Wines with higher alcohol content tend to have significantly higher quality ratings.
# 
# These observations provide insights into the relationships between various wine characteristics and wine quality. For instance, volatile acidity, citric acid, alcohol, and sulphates appear to have notable associations with wine quality. Conversely, residual sugar, free sulfur dioxide, and total sulfur dioxide have relatively weak associations with quality. Understanding these relationships can be valuable for making predictions and recommendations related to wine quality based on these characteristics.

# ## MULTIVARIATE Analysis

# In[33]:


sns.pairplot(df)


# ## Skewness

# In[34]:


#checking Skewness & Kurtosis
print("Skewness in the data \n",df.skew())
print("\nKurtosis in the data \n",df.kurt())


# In[35]:


#lets Visualize it

plt.figure(figsize=(20, 15))
p=1
for i in df.columns[:-1]:
    if p <= 12:
        plt.subplot(4,3,p)
        sns.histplot(data=df, x=i, kde=True)
        plt.xticks()
        plt.title(f'Distribution of {i}')
        plt.xlabel(i)
        p += 1
plt.tight_layout()
plt.show()


# Observation
# 
# - 'Residual sugar' & 'chlorides' variables has high positive skewness. This two variables are skewed to the right, with a longer tail on the positive side.
# - 'Sulphates' has positive skewness with value 2.406505, indicates right skewed distribution.
# - 'Free sulfur dioxide' & 'Total sulfur dioxide' has moderate right Skewed distribution.
# - 'Fixed acidity' and 'Volatile acidity' have positive skewness, although their skewness values are relatively lower at 0.940002 and 0.728474, respectively. This suggests a slight right-skewness in their distributions.
# - 'pH' and 'quality' shows minimal skewness & their distributions are approximately symmetric.
# - 'Citric acid', 'Density', and 'Alcohol' variables have very low skewness values, close to zero. This indicates that their distributions are approximately symmetric or very close to being symmetrical.

# ## Outliers

# In[36]:


#checking for outliers in data

plt.figure(figsize=(20, 15))
p=1
for i in df.columns:
    if p <= 12:
        plt.subplot(4,3,p)
        sns.boxplot(data= df, x=df[i])
        plt.xticks()
        plt.xlabel(i)
        p += 1
plt.tight_layout()
plt.show()


# Observations
# > Outliers are present in all the columns.

# In[37]:


#lets Remove outliers
from scipy.stats import zscore
z=np.abs(zscore(df))
threhold=3

#dropping outliers
df_new=df[(z<3).all(axis=1)]
df_new.shape


# In[38]:


#shape of Old & New DataFrame
print("Old DataFrame",df.shape[0])
print("New DataFrame",df_new.shape[0])
print("Data loss Percentage ", ((df.shape[0]-df_new.shape[0])/df.shape[0])*100)


# Observations
# > Clearly visualize that old dataframe has 1359 rows and in our new dataframe we have 1232 rows, so our data loss percentage is 9.34%
# 
# > In thatcase, instead of removing outliers we are removing Skewness from the data.

# In[39]:


from sklearn.preprocessing import PowerTransformer

pt = PowerTransformer()

df2 = pt.fit_transform(df.iloc[:,:-1])
df2 =pd.DataFrame(df2, columns = df.iloc[:,:-1].columns)
df2.head()


# In[40]:


df2.skew()


# In[41]:


#lets Visualize it

plt.figure(figsize=(20, 15))
p=1
for i in df2.columns[:-1]:
    if p <= 12:
        plt.subplot(4,3,p)
        sns.histplot(data=df, x=i, kde=True)
        plt.xticks()
        plt.title(f'Distribution of {i}')
        plt.xlabel(i)
        p += 1
plt.tight_layout()
plt.show()


# Observations
# > As visualize the skewness is removed from our data.Let's handle the imbalance data in our target variable

# In[42]:


#Adding Target variable in our new dataframe
df2['quality']=df['quality']

df2.columns


# In[43]:


df2.shape


# Observations
# 
# > We can notice that we have added quality column in our new dataFrame. Now we have 1359 rows and 12 columns

# In[44]:


# value_count of quality column/Traget Variable
print('Unique values present in alcohol :', len(df2['quality'].value_counts()))
print(df2['quality'].value_counts())
sns.countplot(x='quality', data=df2)
plt.show()


# **Observations:**
# * As we can see that our target data is imbalanced. and as per the description provided we need to classifying the wine quality as good or bad based on quality Lets Handle it.

# ## Classifying and check the wine quality as good or bad based on quality

# In[45]:


# "Bad" or 0 if quality of wine lies in range of (3 to 6)
# "Good" or 1 if the quality of wine lies in range 7 or above 7

df2['quality']=df2['quality'].apply(lambda x: 1 if x > 6 else 0)
df2.head()


# In[46]:


# Checking current value count of quality variable now
print(df2['quality'].value_counts())
sns.countplot(x='quality', data=df2)
plt.show()


# Observations:
# > As we can clearly visualise the ratio of 0(Bad Quality wine) is very high compared to 1(good quality wine)Lets Handle it.

# In[47]:


x =df2.iloc[:,:-1]
y =df2.iloc[:,-1]
print(x.shape, y.shape)


# ## Applying SMOTE to handle imbalance data in our target column

# In[48]:


from imblearn.over_sampling import SMOTE
sm= SMOTE()
xtrain,ytrain = sm.fit_resample(x,y)
print("Shape of train_x :",xtrain.shape)
print("Shape of train_y :",ytrain.shape)


# In[49]:


# Checking current value count of quality variable now
ytrain.value_counts()


# Observations
# > As we use upscaling(SMOTE) so this increase the no. of samples to improve the accuracy

# In[50]:


import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[51]:


def calc_vif(xtrain):
    vif=pd.DataFrame()
    vif['variables']=xtrain.columns
    vif['VIF FACTOR']=[variance_inflation_factor(xtrain.values,i) for i in range(xtrain.shape[1])]
    return(vif)

calc_vif(xtrain)


# Observations
# 
# > As we can see VIF is less than 10 we will not remove any columns, and proceed further.
# > Since our output is having only two values "0" and "1", we will use binary classification model.

# In[52]:


from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix 


# In[53]:


for i in range(0,1000):
    x_train, x_test, y_train, y_test = train_test_split(xtrain,ytrain,random_state= i, test_size=.20)
    lr=LogisticRegression()
    lr.fit(x_train,y_train)
    pred_train= lr.predict(x_train)
    pred_test=lr.predict(x_test)
    if round(accuracy_score(y_train, pred_train)*100,1)== round(accuracy_score(y_test, pred_test)*100,1):
        print("At Random state",i, "The model perform very well")
        print("At random State:",i)
        print("Training r2_score",accuracy_score(y_train,pred_train)*100)
        print("testing r2 score ",accuracy_score(y_test,pred_test)*100)


# In[54]:


x_train, x_test, y_train, y_test= train_test_split(xtrain,ytrain,random_state=69,test_size=0.20)


# In[55]:


from sklearn.metrics import classification_report
print(classification_report(y_test,pred_test))


# In[56]:


pred_lr = lr.predict(x_test)
from sklearn.model_selection import cross_val_score
lss=accuracy_score(y_test,pred_lr)
for j in range(2,10):
    lsscore= cross_val_score(lr,xtrain,ytrain,cv=j)
    lsc=lsscore.mean()
    print("at cv:-", j)
    print("Cross Validation scre is:-",lsc*100)
    print("Accuracy Score:-", lss*100)
    print("\n")


# In[57]:


lssscore_selected=cross_val_score(lr,xtrain,ytrain,cv=9).mean()
print("The cv score is: ", lssscore_selected,"\nThe accuracy score is: ", lss)


# In[58]:


rfcv=RandomForestClassifier()
rfcv.fit(x_train, y_train)
rfcv.score(x_train,y_train)
rfpred=rfcv.predict(x_test)

print (accuracy_score(y_test,rfpred))
print (confusion_matrix(y_test,rfpred))
print (classification_report(y_test,rfpred))
print('\n')

RFA=accuracy_score(y_test,rfpred)
RFcv=cross_val_score(rfcv,xtrain,ytrain,cv=9).mean()
print("The cv score is: ", RFcv,"\nThe accuracy score is: ", RFA)


# In[59]:


gbcv=GaussianNB()
gbcv.fit(x_train, y_train)
gbcv.score(x_train,y_train)
gbpred=gbcv.predict(x_test)

print (accuracy_score(y_test,gbpred))
print (confusion_matrix(y_test,gbpred))
print (classification_report(y_test,gbpred))
print('\n')

GBA=accuracy_score(y_test,rfpred)
GBcv=cross_val_score(gbcv,xtrain,ytrain,cv=9).mean()
print("The cv score is: ", GBcv,"\nThe accuracy score is: ", GBA)


# In[60]:


kccv=KNeighborsClassifier()
kccv.fit(x_train, y_train)
kccv.score(x_train,y_train)
kcpred=kccv.predict(x_test)

print (accuracy_score(y_test,kcpred))
print (confusion_matrix(y_test,kcpred))
print (classification_report(y_test,kcpred))
print('\n')

KCA=accuracy_score(y_test,kcpred)
KCcv=cross_val_score(kccv,xtrain,ytrain,cv=9).mean()
print("The cv score is: ", KCcv,"\nThe accuracy score is: ", KCA)


# In[61]:


dtcv=DecisionTreeClassifier()
dtcv.fit(x_train, y_train)
dtcv.score(x_train,y_train)
dtpred=dtcv.predict(x_test)

print (accuracy_score(y_test,dtpred))
print (confusion_matrix(y_test,dtpred))
print (classification_report(y_test,dtpred))
print('\n')

DTA=accuracy_score(y_test,dtpred)
DTcv=cross_val_score(dtcv,xtrain,ytrain,cv=9).mean()
print("The cv score is: ", dtcv,"\nThe accuracy score is: ", DTA)


# ### Model Selection

# In[62]:


Model= [LogisticRegression(),RandomForestClassifier(),GaussianNB(),KNeighborsClassifier(),DecisionTreeClassifier(), SVC()]

for m in Model:
    m.fit(x_train,y_train)
    m.score(x_train,y_train)
    predm=m.predict(x_test)
    print('Accuracy score of', m, 'is:')
    print (accuracy_score(y_test,predm))
    print (confusion_matrix(y_test,predm))
    print (classification_report(y_test,predm))
    print('\n')


# ### Hyperparameter tuning: GridSearchCV (Accuracy Check)

# In[63]:


from sklearn.model_selection import GridSearchCV


# In[64]:


parameters= {'n_estimators' : [20, 40, 60, 80, 100,120, 140,160, 200],
             'criterion':['gini', 'entropy'],
             'bootstrap':[0,1,0,1,0,1]}

model=RandomForestClassifier()
Grid= GridSearchCV(estimator=model, param_grid= parameters)
Grid.fit(x_train, y_train)


print(Grid)
print(Grid.best_score_)
print(Grid.best_params_)


# In[65]:


p=Grid.predict(x_test)

print (accuracy_score(y_test,p))
print (confusion_matrix(y_test,p))
print (classification_report(y_test,p))
print('\n')


# Observation
# > As we can see our model accuracy increased from 91% to 93% after hyper tuning

# In[66]:


RF=RandomForestClassifier()
RF.fit(x_train,y_train)
RF.score(x_train,y_train)

RFpred=RF.predict(x_test)
print(accuracy_score(y_test,RFpred ))
print(confusion_matrix(y_test,RFpred ))
print(classification_report(y_test,RFpred ))


# In[67]:


from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds= roc_curve(p, y_test)
roc_auc= auc(fpr, tpr)

plt.figure()
plt.plot(fpr,tpr, color='darkorange', lw=10,label='ROC curve (area= %0.2f)' %roc_auc)
plt.plot([0,1],[0,1],color ='navy', lw=10, linestyle= '--')
plt.xlim([0.0, 1.0])
plt.xlim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()


# # save the Model

# In[81]:


import pickle
filename='Red_Wine_Quality_Prediction.pkl'
pickle.dump(RF,open(filename,'wb'))


# In[69]:


#loading Model
load_model=pickle.load(open('Red_Wine_Quality_Prediction.pkl','rb'))
result=load_model.score(x_test,y_test)
print(result*100)


# In[70]:


conclusion=pd.DataFrame([load_model.predict(x_test)[:],y_test[:]],index=['Predicted','Original'])
conclusion


# In[ ]:




