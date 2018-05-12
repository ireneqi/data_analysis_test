
# coding: utf-8

# In[198]:


import unicodecsv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().magic(u'matplotlib inline')


# #check the data construction

# In[199]:


import unicodecsv
def read_csv(file_name):
    with open(file_name, 'rb') as f:
        reader = unicodecsv.DictReader(f)
        return list(reader)
titanic_data = read_csv('titanic_data.csv')
print titanic_data[0]


# #adj data type 
# #Age lost
# #Cabin lost

# In[200]:


def parse_maybe_float(i):
    if i == '':
        return None
    else:
        return float(i)
    
def parse_maybe_str(i):
    if i == '':
        return None
    else:
        return str(i)
    
for titanic_data_item in titanic_data:
    titanic_data_item['Age'] = parse_maybe_float(titanic_data_item['Age'])
    titanic_data_item['Cabin'] = parse_maybe_str(titanic_data_item['Cabin'])
titanic_data[0]


# In[201]:


len(titanic_data)


# In[202]:


titanic_df=pd.read_csv('titanic_data.csv')


# In[203]:


titanic_df.head()


# #Age Cabin Embarked lost some  data

# In[204]:


titanic_df.info()


# #Survived% is only 38.38%

# In[205]:


titanic_df.describe()


# In[206]:


#count the survived and no survived Passenger
num_survived = titanic_df['Survived'].sum()
num_no_survived = 891 - num_survived

print num_survived
print num_no_survived


# #draw a Figure  to show Survived and Victim

# In[259]:



plt.figure(figsize = (12,6))
plt.subplot(121)
sns.countplot(x='Survived', data=titanic_df)
plt.title('Survived person count')

plt.subplot(122)
plt.pie([num_survived, num_no_survived],labels=['Survived','Victim'],autopct='%1.0f%%')
plt.title('Survived persontage') 

plt.show()


# In[260]:


#according to the film, I prefer to check the persentage of female survived 
#step 1  I should check how many people on Titanic before wrecked
male_sum = titanic_df['Sex'][titanic_df['Sex'] == 'male'].count()
female_sum = titanic_df['Sex'][titanic_df['Sex'] == 'female'].count()
print male_sum
print female_sum


# In[261]:


plt.figure(figsize=(12,6))
plt.subplot(121)
sns.countplot(x='Sex', data=titanic_df)
plt.subplot(122)
plt.pie([male_sum,female_sum],labels=['male', 'female'],autopct='%1.0f%%')
plt.show()


# #there are 577 male, and 314 female before wrecked

# In[209]:


survived_df = titanic_df[titanic_df[ 'Survived'] == 1 ]


# #after the Titanic wreck
# # check the Survived of male and female

# In[262]:


survived_male_sum = survived_df['Sex'][survived_df['Sex'] == 'male'].count()
survived_female_sum = survived_df['Sex'][survived_df['Sex'] == 'female'].count()
print survived_male_sum 
print survived_female_sum


# In[266]:


plt.figure(figsize=(10,5))
plt.subplot(121)
sns.countplot(x='Sex', data=survived_df)
plt.subplot(122)
plt.pie([survived_male_sum, survived_female_sum],labels=['survived male', 'survived female'],autopct='%1.0f%%')
plt.show()


# #the survived female 109 is more than male 233,

# In[264]:



male_df = titanic_df[titanic_df['Sex'] == 'male']

plt.figure(figsize=(10,5))
plt.subplot(121)
sns.countplot(x = 'Survived', data = male_df)
plt.subplot(122)
plt.pie([male_df['Survived'][male_df['Survived'] == 0].count(),male_df['Survived'][male_df['Survived'] == 1].count()],labels=['Male Victim', 'Survived male'],autopct='%1.0f%%')
plt.show()


# #Survived persentage of male

# In[265]:


female_df = titanic_df[titanic_df['Sex'] == 'female']

plt.figure(figsize=(10,5))
plt.subplot(121)
sns.countplot(x = 'Survived', data = female_df)
plt.subplot(122)
plt.pie([female_df['Survived'][female_df['Survived'] == 0].count(),female_df['Survived'][female_df['Survived'] == 1].count()],labels=['Victim female', 'Survived female'],autopct='%1.0f%%')
plt.show()



#  #the survived female 74% percentage is higher than male 19%， Titanic film tells us the true,from the result above.

# # Secondly,I prefre to check the Survived age. as the Titanic Film.
# # I found the some of the age data is lost; we need to fill in  the lost age of data.

# In[214]:


avg_age_titanic   = titanic_df["Age"].mean()
std_age_titanic   = titanic_df["Age"].std()
count_nan_age_titanic = titanic_df["Age"].isnull().sum()


rand_age = np.random.randint(avg_age_titanic - std_age_titanic, avg_age_titanic + std_age_titanic, size = count_nan_age_titanic)

titanic_df["Age"][np.isnan(titanic_df["Age"])] = rand_age


# #random age，range (mean - std， mean + std)

# In[215]:


plt.figure(figsize=(12,5))
plt.subplot(121)
titanic_df['Age'].hist(bins = 70)
plt.xlabel('Age')
plt.ylabel('Num')

plt.subplot(122)
titanic_df.boxplot(column='Age', showfliers=False)

plt.show()


# #acccording to the hist, I found the age data Concentrated in the peak age between 20-40 years of age，the boxplot may show the result  visually。

# In[267]:


titanic_df['Age'].describe()


# #According to the sample,total 891, the avg is  29.59, std  13.54,the youngest is 0.42, the eldest is 80.

# # before analysis the age result,I group the age data to children,Teenagers，adult，and elderly people 4 groups

# In[269]:


bins = [0, 12, 18, 65, 100]
titanic_df['Age_group'] = pd.cut(titanic_df['Age'], bins)
by_age = titanic_df.groupby('Age_group')['Survived'].mean()
by_age


# #the Survived persontage of children is higher than Teenagers and adult, the elderly people  is lowest.

# In[218]:


by_age.plot(kind = "bar")


# #Survived persentage by different Pclass

# In[219]:


titanic_df[['Pclass','Survived']].groupby(['Pclass']).count()


# #analysis data from the Pclass before the Ship wreck

# In[270]:


plt.figure(figsize= (10 ,5))
plt.subplot(121)
sns.countplot(x='Pclass', data=titanic_df)
plt.title('Pclass Count') 

plt.subplot(122)
plt.pie(titanic_df[['Pclass','Survived']].groupby(['Pclass']).count(),labels=['Pclass_1','Pclass_2','Pclass_3'],autopct='%1.0f%%')

plt.show()


# In[271]:


survived_df[['Pclass','Survived']].groupby(['Pclass']).sum()


# # analysis data from the Pclass before the Ship wreck

# In[273]:


plt.figure(figsize= (10, 5))
plt.subplot(121)
sns.countplot(x='Pclass', data=survived_df)
plt.title('Survived by Pclass') 
plt.ylabel('Survived Count')

plt.subplot(122)
plt.pie(survived_df[['Pclass','Survived']].groupby(['Pclass']).sum(),labels=['Pclass_1','Pclass_2','Pclass_3'],autopct='%1.0f%%')
plt.show()


# In[274]:


bins = [0 ,1, 2, 3]
titanic_df['Pclass_group'] = pd.cut(titanic_df['Pclass'], bins)
by_Pclass = titanic_df.groupby('Pclass_group')['Survived'].mean()
by_Pclass
# the percentage and count of Survived from Pclass_1 is highest


# In[275]:


by_Pclass.plot(kind = "bar")


# In[225]:


#Survived by Fare
plt.figure(figsize=(10,5))
titanic_df['Fare'].hist(bins = 50)

titanic_df.boxplot(column='Fare', by='Survived', showfliers=False)
plt.show()


# #the Survived Passenger in higher Fare is higher than lower Fare

# # check the data lost Embarked

# In[276]:


num_lost_Embarked = 0
for titanic_data_item in titanic_data:

    if(titanic_data_item['Embarked']==''):
        print titanic_data_item
        num_lost_Embarked += 1

num_lost_Embarked


# #lost the Embarked of 2 and  u'Ticket is all 113572,the others 1135**'s Embarked is S，and mode is also S

# In[227]:


titanic_df["Embarked"] = titanic_df["Embarked"].fillna("S")


# In[277]:


fig, (axis1,axis2,axis3) = plt.subplots(1,3,figsize=(15,5)) 

sns.countplot(x='Embarked', data=titanic_df, ax=axis1)
embark_perc = titanic_df[["Embarked", "Survived"]].groupby(['Embarked'],as_index=False).mean()
sns.barplot(x='Embarked', y='Survived', data=embark_perc,order=['S','C','Q'],ax=axis2)
sns.countplot(x='Survived', hue="Embarked", data=titanic_df, order=[1,0], ax=axis3)


# # Survived by have sibling or not

# In[279]:


sibsp_df = titanic_df[titanic_df['SibSp'] != 0]
no_sibsp_df = titanic_df[titanic_df['SibSp'] == 0]


# #Survived passager in Embarked S is largest，EmbarkedS Q is Minimum。Survived percentage in Embarked C is highest and S is lowest。

# In[230]:


#the psassageer
plt.figure(figsize=(10,5))
plt.subplot(121)
sns.countplot(x = 'Survived', data = sibsp_df )

plt.subplot(122)
plt.pie([sibsp_df['Survived'][sibsp_df['Survived'] == 0].count(),sibsp_df['Survived'].sum()],labels=['Victim', 'Survived'],autopct='%1.0f%%')
plt.show()
print sibsp_df['Survived'][sibsp_df['Survived'] == 0].count(), sibsp_df['Survived'].sum()


# #the count of survived passagers have siblings is 132, survived percentage is 47%,higher than the average 38%

# In[231]:


plt.figure(figsize=(10,5))
plt.subplot(121)
sns.countplot(x = 'Survived', data = no_sibsp_df )

plt.subplot(122)
plt.pie([no_sibsp_df['Survived'][no_sibsp_df['Survived'] == 0].count(),no_sibsp_df['Survived'].sum()],labels=['Victim', 'Survived'],autopct='%1.0f%%')
plt.show()
print no_sibsp_df['Survived'][no_sibsp_df['Survived'] == 0].count(), no_sibsp_df['Survived'].sum()
#the count of Survived passsage have no siblings is 210, percentage is 35%,lower than the passsage have siblings.


# In[232]:


# Survived by Parch
parch_df = titanic_df[titanic_df['Parch'] != 0]
no_parch_df = titanic_df[titanic_df['Parch'] == 0]


# In[280]:


plt.figure(figsize=(10,5))
plt.subplot(121)
sns.countplot(x = 'Survived', data = parch_df )

plt.subplot(122)
plt.pie([parch_df['Survived'][parch_df['Survived'] == 0].count(),parch_df['Survived'].sum()],labels=['Victim', 'Survived'],autopct='%1.0f%%')
plt.show()
print parch_df['Survived'][parch_df['Survived'] == 0].count(),parch_df['Survived'].sum()


# the count of survived passagers have parent is 109, and the survived percentage is 51%

# In[234]:


plt.figure(figsize=(10,5))
plt.subplot(121)
sns.countplot(x = 'Survived', data = no_parch_df)

plt.subplot(122)
plt.pie([no_parch_df['Survived'][no_parch_df['Survived'] == 0].count(),no_parch_df['Survived'].sum()],labels=['Victim', 'Survived'],autopct='%1.0f%%')
plt.show()
print no_parch_df['Survived'][no_parch_df['Survived'] == 0].count(),no_parch_df['Survived'].sum()


# #the count of survived passagers have no parent is 233, and the survived percentage is 34%
# #the survived percentage of passengers have parent together is highter than passengers have no parent together 

# In[281]:


data_by_location = titanic_df.groupby(['Sex','Age'],as_index=False).mean()


# In[282]:


data_by_location.head()


# In[237]:


data_by_location.head()['Age']


# In[238]:


get_ipython().magic(u'pylab inline')
import matplotlib.pyplot as plt
import seaborn as sns


# In[239]:


scaled_entries_1 = (data_by_location['Survived']/data_by_location['Survived'].std())


# In[240]:


plt.scatter(data_by_location['Sex'],data_by_location['Age'],s=scaled_entries_1)


# #female survived percentage is higher than male

# In[283]:


data_by_location = titanic_df.groupby(['Pclass','Age'],as_index=False).mean()


# In[284]:


data_by_location.head()


# In[243]:


data_by_location.head()['Age']


# In[244]:


get_ipython().magic(u'pylab inline')
import matplotlib.pyplot as plt
import seaborn as sns


# In[245]:


scaled_entries_2 = (data_by_location['Survived']/data_by_location['Survived'].std())


# In[246]:


plt.scatter(data_by_location['Pclass'],data_by_location['Age'],s=scaled_entries_2)


# In[285]:


data_by_location = titanic_df.groupby(['Fare','Age'],as_index=False).mean()


# In[286]:


data_by_location.head()


# In[287]:


data_by_location.head()['Age']


# In[288]:


get_ipython().magic(u'pylab inline')
import matplotlib.pyplot as plt
import seaborn as sns


# In[289]:


scaled_entries = (data_by_location['Survived']/data_by_location['Survived'].std())


# In[290]:


plt.scatter(data_by_location['Fare'],data_by_location['Age'],s=scaled_entries)


# #通过绘图得知在泰坦尼克船上，获救的人集中在Fare较低（0,50]，年龄集中在10岁以上35岁以下乘客中的几率比较大

# In[292]:


data_by_location = titanic_df.groupby(['Parch','SibSp'],as_index=False).mean()


# In[293]:


data_by_location.head()


# In[255]:


data_by_location.head()['SibSp']


# In[256]:


get_ipython().magic(u'pylab inline')
import matplotlib.pyplot as plt
import seaborn as sns


# In[257]:


plt.scatter(data_by_location['SibSp'],data_by_location['Parch'],s=scaled_entries)


# #no parents or siblings passagers survived percentage is lower.

# # inconclusion, the female passenage in  Pclass _1, who have child,siblings parents on ship, age yonger, have higher persentge of Survived
