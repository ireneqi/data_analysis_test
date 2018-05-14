
# coding: utf-8

# # Question(提出问题)

# According to the film <Titanic>, I remember that the main plot while escaping，as ship accident occurred , the female children first, and male after them. 
# 

# Question：Is the plot of the film responding to what actually happened just like lady first?

# Question：Is the age will affect Survived.

# question : Any other influencing factors?

# # Wrangle(数据整理)

# In[19]:


import unicodecsv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().magic(u'matplotlib inline')


# check the data construction

# In[20]:


# import unicodecsv
# def read_csv(file_name):
#     with open(file_name, 'rb') as f:
#         reader = unicodecsv.DictReader(f)
#         return list(reader)
# titanic_data = read_csv('titanic_data.csv')
# print titanic_data[0]


# In[21]:


titanic_df=pd.read_csv('titanic_data.csv')


# In[22]:


titanic_df.head()


# #Age Cabin Embarked lost some  data

# In[23]:


titanic_df.info()


# # 从以上结果可以看出，样本数量为891，Age Canin Embarked 存在缺失

# Survived：是否存活（0代表否，1代表是）
# 
# Pclass：船舱等级（1代表一等舱，2代表二等舱，3代表三等舱）
# 
# Name：船上乘客的名字
# 
# Sex：船上乘客的性别
# 
# # Age：船上乘客的年龄（存在数据缺失）见下面代码 random age，range (mean - std， mean + std),在这个区域随机生成补充数据
# 
# SibSp：乘客在船上的兄弟姐妹和配偶的数量
# 
# Parch：乘客在船上的父母以及小孩的数量
# 
# Ticket：乘客船票的编号
# 
# Fare：乘客为船票支付的费用
# 
# Cabin：乘客所在船舱的编号
# 
# # Embarked：乘客上船的港口（C 代表从 Cherbourg 登船，Q 代表从 Queenstown 登船，S 代表从 Southampton 登船）（存在三个缺失，用众数填充）
# 
# 根据电影情节主要研究不同因素对乘客获救与否的影响Sex、Age、Pclass影响重点研究，SibSp、Parch、Fare作为次重点研究，Name，Ticket，Cabin， PassengerId影响不具代表性 不做研究。
# 
# 缺失数据在研究时根据众数，平均值，偏差填充

# # Explore(数据探索)

# Find the average of Survived % from describe all item of titanic_df

# In[24]:


titanic_df.describe()


# Survived% is only 38.38%

# # adj Count the survived and Victim Passenger function

# In[25]:


num_survived = titanic_df['Survived'].sum()
num_no_survived = titanic_df['PassengerId'].count() - num_survived

print num_survived
print num_no_survived


# draw a Figure  to show Survived and Victim

# In[26]:


plt.figure(figsize = (12,6))
plt.subplot(121)
sns.countplot(x='Survived', data=titanic_df)
plt.title('Survived person count')

plt.subplot(122)
plt.pie([num_survived, num_no_survived],labels=['Survived','Victim'],autopct='%1.0f%%')
plt.title('Survived persontage') 

plt.show()


# # Firstly According to the film, I prefer to check the persentage of female survived 
# 
# step 1  I should check how many male and female on Titanic before wrecked

# In[27]:


male_sum = titanic_df['Sex'][titanic_df['Sex'] == 'male'].count()
female_sum = titanic_df['Sex'][titanic_df['Sex'] == 'female'].count()
print male_sum
print female_sum


# In[28]:


plt.figure(figsize=(12,6))
plt.subplot(121)
sns.countplot(x='Sex', data=titanic_df)
plt.subplot(122)
plt.pie([male_sum,female_sum],labels=['male', 'female'],autopct='%1.0f%%')
plt.show()


# There are 577 male, and 314 female before wrecked
# 
# step 2  I should check how many male and female on Titanic after wrecked

# In[29]:


survived_df = titanic_df[titanic_df[ 'Survived'] == 1 ]


# In[30]:


survived_male_sum = survived_df['Sex'][survived_df['Sex'] == 'male'].count()
survived_female_sum = survived_df['Sex'][survived_df['Sex'] == 'female'].count()
print survived_male_sum 
print survived_female_sum


# In[31]:


plt.figure(figsize=(10,5))
plt.subplot(121)
sns.countplot(x='Sex', data=survived_df)
plt.subplot(122)
plt.pie([survived_male_sum, survived_female_sum],labels=['survived male', 'survived female'],autopct='%1.0f%%')
plt.show()


# The survived female 109 is more than male 233,

# In[32]:


male_df = titanic_df[titanic_df['Sex'] == 'male']

plt.figure(figsize=(10,5))
plt.subplot(121)
sns.countplot(x = 'Survived', data = male_df)
plt.subplot(122)
plt.pie([male_df['Survived'][male_df['Survived'] == 0].count(),male_df['Survived'][male_df['Survived'] == 1].count()],labels=['Male Victim', 'Survived male'],autopct='%1.0f%%')
plt.show()


# #Survived persentage of male

# In[33]:


female_df = titanic_df[titanic_df['Sex'] == 'female']

plt.figure(figsize=(10,5))
plt.subplot(121)
sns.countplot(x = 'Survived', data = female_df)
plt.subplot(122)
plt.pie([female_df['Survived'][female_df['Survived'] == 0].count(),female_df['Survived'][female_df['Survived'] == 1].count()],labels=['Victim female', 'Survived female'],autopct='%1.0f%%')
plt.show()



#  #the survived female 74% percentage is higher than male 19%， Titanic film tells us the true,from the result above.

# #Secondly,I prefre to check the Survived age. as the Titanic Film.
# # I found the some of the age data is lost; we need to fill in  the lost age of data.

# In[34]:


avg_age_titanic   = titanic_df["Age"].mean()
std_age_titanic   = titanic_df["Age"].std()
count_nan_age_titanic = titanic_df["Age"].isnull().sum()


rand_age = np.random.randint(avg_age_titanic - std_age_titanic, avg_age_titanic + std_age_titanic, size = count_nan_age_titanic)

titanic_df["Age"][np.isnan(titanic_df["Age"])] = rand_age


# #random age，range (mean - std， mean + std)

# In[35]:


plt.figure(figsize=(12,5))
plt.subplot(121)
titanic_df['Age'].hist(bins = 70)
plt.xlabel('Age')
plt.ylabel('Num')

plt.subplot(122)
titanic_df.boxplot(column='Age', showfliers=False)

plt.show()


# #acccording to the hist, I found the age data Concentrated in the peak age between 20-40 years of age，the boxplot may show the result  visually。

# In[36]:


titanic_df['Age'].describe()


# #According to the sample,total 891, the avg is  29.59, std  13.54,the youngest is 0.42, the eldest is 80.

# # before analysis the age result,I group the age data to children,Teenagers，adult，and elderly people 4 groups

# In[88]:


print titanic_df['Age_group'].value_counts().sort_index() # 首先, 研究组内人数
titanic_df.groupby('Age_group')['Survived'].mean().plot(kind='bar', color='purple', title='Age_group vs Survival', ylim=(0, 1)) # 然后, 展示组内生还率


# Thanks for reviewer's suggestion:)

# In[86]:


bins = [0, 12, 18, 65, 100]
titanic_df['Age_group'] = pd.cut(titanic_df['Age'], bins)
by_age = titanic_df.groupby('Age_group')['Survived'].mean()
by_age


# #the Survived persontage of children is higher than Teenagers and adult, the elderly people  is lowest.

# In[87]:


by_age.plot(kind = "bar")


# #Survived persentage by different Pclass

# In[39]:


titanic_df[['Pclass','Survived']].groupby(['Pclass']).count()


# #  Thirdly, analysis data from the Pclass before the Ship wreck

# In[40]:


plt.figure(figsize= (10 ,5))
plt.subplot(121)
sns.countplot(x='Pclass', data=titanic_df)
plt.title('Pclass Count') 

plt.subplot(122)
plt.pie(titanic_df[['Pclass','Survived']].groupby(['Pclass']).count(),labels=['Pclass_1','Pclass_2','Pclass_3'],autopct='%1.0f%%')

plt.show()


# In[41]:


survived_df[['Pclass','Survived']].groupby(['Pclass']).sum()


# In[42]:


plt.figure(figsize= (10, 5))
plt.subplot(121)
sns.countplot(x='Pclass', data=survived_df)
plt.title('Survived by Pclass') 
plt.ylabel('Survived Count')

plt.subplot(122)
plt.pie(survived_df[['Pclass','Survived']].groupby(['Pclass']).sum(),labels=['Pclass_1','Pclass_2','Pclass_3'],autopct='%1.0f%%')
plt.show()


# In[43]:


bins = [0 ,1, 2, 3]
titanic_df['Pclass_group'] = pd.cut(titanic_df['Pclass'], bins)
by_Pclass = titanic_df.groupby('Pclass_group')['Survived'].mean()
by_Pclass
# the percentage and count of Survived from Pclass_1 is highest


# In[44]:


by_Pclass.plot(kind = "bar")


# In[45]:


#Survived by Fare
plt.figure(figsize=(10,5))
titanic_df['Fare'].hist(bins = 50)

titanic_df.boxplot(column='Fare', by='Survived', showfliers=False)
plt.show()


# #the Survived Passenger in higher Fare is higher than lower Fare

# # check the data lost Embarked

# In[46]:


titanic_df[titanic_df.Embarked.isnull()]


# # lost the Embarked of 2 and  u'Ticket is all 113572,the others 1135**'s Embarked is S，and mode is also S

# In[47]:


titanic_df["Embarked"] = titanic_df["Embarked"].fillna("S")


# In[93]:


fig, (axis1,axis2,axis3) = plt.subplots(1,3,figsize=(15,5)) 

sns.countplot(x='Embarked', data=titanic_df, ax=axis1)
embark_perc = titanic_df[["Embarked", "Survived"]].groupby(['Embarked'],as_index=False).mean()
sns.barplot(x='Embarked', y='Survived', data=embark_perc,order=['S','C','Q'],ax=axis2)
sns.countplot(x='Survived', hue="Embarked", data=titanic_df, order=[1,0], ax=axis3)


# 1.The Picture left shows the count of the people embarked from the three port （C 代表从 Cherbourg 登船，Q 代表从 Queenstown 登船，S 代表从 Southampton 登船）,the number people embarked from Southampton is largest, and  the people from Queenstown is least.
# 
# 2.The Survived percentage from hight to low is C>Q>S shows by the Picture center.
# 
# 3.The right Picture shows the compare result about Survived(1) and unSurvived(0).
# 
# According to the three pictures, the people embarked Cherbourg most likely to be rescued. Maybe the seat for passengers embarked form Cherbourg close to the escape boat, or some other reasons.

# # Survived by have sibling or not

# In[49]:


sibsp_df = titanic_df[titanic_df['SibSp'] != 0]
no_sibsp_df = titanic_df[titanic_df['SibSp'] == 0]


# #Survived passager in Embarked S is largest，EmbarkedS Q is Minimum。Survived percentage in Embarked C is highest and S is lowest。

# In[50]:


#the psassageer
plt.figure(figsize=(10,5))
plt.subplot(121)
sns.countplot(x = 'Survived', data = sibsp_df )

plt.subplot(122)
plt.pie([sibsp_df['Survived'][sibsp_df['Survived'] == 0].count(),sibsp_df['Survived'].sum()],labels=['Victim', 'Survived'],autopct='%1.0f%%')
plt.show()
print sibsp_df['Survived'][sibsp_df['Survived'] == 0].count(), sibsp_df['Survived'].sum()


# The count of survived passagers have siblings is 132, survived percentage is 47%,higher than the average 38%, the passengers have siblings may help each other to survival.

# In[51]:


plt.figure(figsize=(10,5))
plt.subplot(121)
sns.countplot(x = 'Survived', data = no_sibsp_df )

plt.subplot(122)
plt.pie([no_sibsp_df['Survived'][no_sibsp_df['Survived'] == 0].count(),no_sibsp_df['Survived'].sum()],labels=['Victim', 'Survived'],autopct='%1.0f%%')
plt.show()
print no_sibsp_df['Survived'][no_sibsp_df['Survived'] == 0].count(), no_sibsp_df['Survived'].sum()
#the count of Survived passsage have no siblings is 210, percentage is 35%,lower than the passsage have siblings.


# In[52]:


# Survived by Parch
parch_df = titanic_df[titanic_df['Parch'] != 0]
no_parch_df = titanic_df[titanic_df['Parch'] == 0]


# In[53]:


plt.figure(figsize=(10,5))
plt.subplot(121)
sns.countplot(x = 'Survived', data = parch_df )

plt.subplot(122)
plt.pie([parch_df['Survived'][parch_df['Survived'] == 0].count(),parch_df['Survived'].sum()],labels=['Victim', 'Survived'],autopct='%1.0f%%')
plt.show()
print parch_df['Survived'][parch_df['Survived'] == 0].count(),parch_df['Survived'].sum()


# the count of survived passagers have parent is 109, and the survived percentage is 51%, higher than the average 38%, the passengers have parents or children, likely to be mother and chlidren, the result may verified from another factors, female may have priority to be survival.

# In[54]:


plt.figure(figsize=(10,5))
plt.subplot(121)
sns.countplot(x = 'Survived', data = no_parch_df)

plt.subplot(122)
plt.pie([no_parch_df['Survived'][no_parch_df['Survived'] == 0].count(),no_parch_df['Survived'].sum()],labels=['Victim', 'Survived'],autopct='%1.0f%%')
plt.show()
print no_parch_df['Survived'][no_parch_df['Survived'] == 0].count(),no_parch_df['Survived'].sum()


# #the count of survived passagers have no parent is 233, and the survived percentage is 34%
# #the survived percentage of passengers have parent together is highter than passengers have no parent together 

# # Draw Conclusions(得出结论)

# To analyst the Survived result in Titanic data,  The main influencing factors age and sex I assumed at beginning,seems effected to the result.however the result is limited to the sample only, the whole data of Titanic is much more than 891.
# 
# 1.Sex, the Male passenagers at begining is 577 ,and female is 314;after the ship crash,the survived female 109 is more than male 233;
# 
# 2.Age, before analysis the age result,I group the age data to children,Teenagers，adult，and elderly people 4 groups,the Survived persontage of children is higher than Teenagers and adult, the elderly people is lowest.
# 
# 3.Pclass, the three Pclass from 1 to 3 are 216 184 491, after the ship crash, the three Pclass from 1 to 3 are 136 87 119.the Survived persentage from high to low is 1>2>3.
# 
# 4. other factors like the siblings SibSp Parch not effect the result much.
# 
# As we check the sample only, we can see the relashionship between the Sex, Age, Pclass factors and Survived, only can result the Correlation， not the Absolute cause and effect.

# # Communicate(交流)

# In[55]:


data_by_location = titanic_df.groupby(['Sex','Age'],as_index=False).mean()


# In[56]:


data_by_location.head()


# In[57]:


data_by_location.head()['Age']


# In[58]:


get_ipython().magic(u'pylab inline')
import matplotlib.pyplot as plt
import seaborn as sns


# In[59]:


scaled_entries_1 = (data_by_location['Survived']/data_by_location['Survived'].std())


# In[60]:


plt.scatter(data_by_location['Sex'],data_by_location['Age'],s=scaled_entries_1)
plt.xlabel('Sex')
plt.ylabel('Age')


# #female survived percentage is higher than male

# In[61]:


data_by_location = titanic_df.groupby(['Pclass','Age'],as_index=False).mean()


# In[62]:


data_by_location.head()


# In[63]:


data_by_location.head()['Age']


# In[64]:


get_ipython().magic(u'pylab inline')
import matplotlib.pyplot as plt
import seaborn as sns


# In[65]:


scaled_entries_2 = (data_by_location['Survived']/data_by_location['Survived'].std())


# In[66]:


plt.scatter(data_by_location['Pclass'],data_by_location['Age'],s=scaled_entries_2)
plt.xlabel('Age')
plt.ylabel('Pclass')


# In[67]:


data_by_location = titanic_df.groupby(['Fare','Age'],as_index=False).mean()


# In[68]:


data_by_location.head()


# In[69]:


data_by_location.head()['Age']


# In[70]:


get_ipython().magic(u'pylab inline')
import matplotlib.pyplot as plt
import seaborn as sns


# In[71]:


scaled_entries = (data_by_location['Survived']/data_by_location['Survived'].std())


# In[72]:


plt.scatter(data_by_location['Fare'],data_by_location['Age'],s=scaled_entries)
plt.xlabel('Fare')
plt.ylabel('Age')


# #通过绘图得知在泰坦尼克船上，获救的人集中在Fare较低（0,50]，年龄集中在10岁以上35岁以下乘客中的几率比较大

# In[73]:


data_by_location = titanic_df.groupby(['Parch','SibSp'],as_index=False).mean()


# In[74]:


data_by_location.head()


# In[75]:


data_by_location.head()['SibSp']


# In[76]:


get_ipython().magic(u'pylab inline')
import matplotlib.pyplot as plt
import seaborn as sns


# In[77]:


plt.scatter(data_by_location['SibSp'],data_by_location['Parch'],s=scaled_entries)
plt.xlabel('SibSp')
plt.ylabel('Parch')


# #no parents or siblings passagers survived percentage is lower.
