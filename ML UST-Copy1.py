#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn import linear_model


# In[6]:


var=pd.read_csv("C:/Users/DELL/Downloads/gapminder-FiveYearData (1).csv")
var


# In[7]:


var


# In[16]:


arr=np.array([1,2,3,4,5,6,5])


# In[17]:


a=np.array([9,10,11,12,13,14,15])


# In[18]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[19]:


import matplotlib.pyplot as plt


# In[24]:


plt.plot(arr,a)
plt.show('year')
plt.


# In[22]:


import matplotlib.pyplot as plt


# In[23]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[25]:


plt.xlabel('year')
plt.ylabel('population')


# In[30]:


plt.scatter(var.gdpPercap,var.pop,color='red',marker='*')


# In[32]:


len(pop)


# In[36]:


np.shape(var['pop'])[0]


# In[37]:


np.shape(var['year'])[0]


# In[38]:


np.shape(var['gdpPercap'])[0]


# In[39]:


plt.scatter(var.year,var.pop,color='red',marker='*')


# In[40]:


var


# In[41]:


var


# In[42]:


vari=pd.read_csv("C:/Users/DELL/Downloads/gapminder-FiveYearData (1).csv")


# In[43]:


vari


# In[44]:


plt.scatter(vari.area,vari.price,color='red',marker='*')


# In[61]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import linear_model


# In[62]:


plt.xlabel('area(sq.ft)')
plt.ylabel('price(rupees)')


# In[63]:


plt.scatter(vari.area,vari.price,color='red',marker='*')


# In[64]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.xlabel('area(sq.ft)')
plt.ylabel('price(rupees)')
plt.scatter(vari.area,vari.price,color='red',marker='*')


# In[65]:


vari


# In[66]:


vari=pd.read_csv("C:/Users/DELL/Downloads/gapminder-FiveYearData (1).csv")


# In[67]:


vari


# In[68]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.xlabel('area(sq.ft)')
plt.ylabel('price(rupees)')
plt.scatter(vari.area,vari.price,color='red',marker='*')


# In[71]:


reg=linear_model.LinearRegression()
reg.fit(vari[['area']],vari.price)


# In[85]:


reg.predict([[2302]])[0]


# In[76]:


reg.coef_


# In[82]:


reg.intercept_


# In[83]:


70*2302+1370000.0


# In[137]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import linear_model


# In[138]:


var2=pd.read_csv("C:/Users/DELL/Downloads/gapminder-FiveYearData (1).csv")


# In[139]:


var2


# In[140]:


plt.xlabel('area(sq.ft)')
plt.ylabel('price(rupees)')
plt.scatter(vari.area,vari.price,color='red',marker='*')


# In[141]:


var2


# In[142]:


var3=pd.read_csv("C:/Users/DELL/Downloads/gapminder-FiveYearData (1).csv")


# In[143]:


var3


# In[144]:


plt.xlabel('area(sq.ft)')
plt.ylabel('price(rupees)')


plt.scatter(var3.area,var3.price,color='red',marker='*')


# In[170]:


ar=var3.area.values.reshape(len('area'),1)
pr=var3.price.values.reshape(len('price')-1,1)
lreg=linear_model.LinearRegression()
lreg.fit(ar,pr)


# In[171]:


lreg.predict(15033)


# In[163]:


lreg.coef_


# In[164]:


lreg.intercept_


# In[165]:



0.7*15033+13700


# In[126]:


var3


# In[156]:


x=var3['area']
x=var3.reshape(len(x),1)


# In[175]:


q=np.array([1,2,3,4,5])
r=np.array([6,7,8,9,0])


# In[176]:


lr=linear_model.LinearRegression()


# In[178]:


lr.fit([[q,r]])


# In[184]:


q=q.reshape(-1,1)
r=r.reshape(-1,1)


# In[185]:


lr.fit(q,r)


# In[188]:


r.shape
q.shape


# In[194]:


plt.xlabel('P')
plt.ylabel('Q')
plt.scatter(q,r,color='blue')


# In[199]:


lr.predict([[5.09]])


# In[200]:


var3


# In[21]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import linear_model


# In[22]:


variable=pd.read_csv("C:/Users/DELL/Downloads/gapminder-FiveYearData (1).csv")


# In[23]:


variable


# In[24]:


x=variable['area']


# In[25]:


x.shape


# In[26]:


x=x.reshape(-1,1)


# In[27]:


x=variable.price.reshape((-1,1))


# In[48]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.xlabel('area(in sq.m)')
plt.ylabel('price(in RS)')
z=plt.scatter(variable.area,variable.price,color='blue',marker='+')


# In[29]:


x=variable.area


# In[30]:


y=variable.price


# In[62]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.xlabel('area(in sq.m)')
plt.ylabel('price(in RS)')
plt.scatter(x,y,color='blue',marker='+')
plt.plot(x,reg_obj.predict(variable[['area']]))


# In[32]:


reg_obj=linear_model.LinearRegression()
reg_obj.fit(variable[['area']],variable.price)


# In[33]:


reg_obj.predict([[2013]])


# In[34]:


reg_obj.coef_


# In[35]:


reg_obj.intercept_


# In[36]:


reg_obj.predict([[6000]])


# In[37]:


predictions=pd.read_csv("C:/Users/DELL/Downloads/prediction.csv")


# In[38]:


predictions


# In[39]:


m=reg_obj.predict(predictions)


# In[40]:


predictions['price']=m


# In[41]:


predictions.to_csv("prediction.csv",index=False)


# In[42]:


predictions.to_csv("prediction.csv")


# In[44]:


predictions


# In[52]:


plt.show()


# In[58]:


plt.scatter(x,y)
plt.plot(variable.area,reg_obj.predict(variable[['area']]),color='blue')


# In[54]:


plt.plot(x,y,color='red')


# In[57]:


plt.plot(variable.area,reg_obj.predict(variable[['area']]),color='blue')


# In[64]:


(/exersice//)


# In[70]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn import linear_model


# In[71]:


exer=pd.read_csv("C:/Users/DELL/Downloads/canada_per_capita_income.csv")


# In[73]:


exer.head(6)


# In[74]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.xlabel('per capita income(US$)')
plt.ylabel('year')
plt.scatter()


# In[76]:


x=exer['per capita income (US$)']


# In[77]:


y=exer['year']


# In[79]:


matplotlib inline
plt.xlabel('per capita income(US$)')
plt.ylabel('year')
plt.scatter(x,y,color='blue')%


# In[93]:


reg_model=linear_model.LinearRegression
reg_model.fit(exer[['year']], exer['per capita income (US$)'])


# In[83]:





# In[94]:


exer


# In[95]:


exer=pd.read_csv("C:/Users/DELL/Downloads/canada_per_capita_income.csv")


# In[96]:


exer


# In[97]:


x=exer['percapitainc']


# In[98]:


y=exer['year']


# In[99]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.xlabel('per capita income(US$)')
plt.ylabel('year')
plt.scatter(x,y,color='blue')


# In[101]:


reg_model=linear_model.LinearRegression()
reg_model.fit(exer[['year']], exer.percapitainc)

reg_model.predict([[2003]])
# In[102]:


reg_model.predict([[2003]])


# In[103]:


reg_obj.coef_


# In[104]:


reg_obj.intercept_


# In[105]:


2003*0.7+13700


# In[106]:


reg_model.predict([[2020]])


# In[109]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.xlabel('per capita income(US$)')
plt.ylabel('year')
plt.scatter(x,y,color='blue')
 plt.plot(x,reg_model.predict(exer[['percapitainc']]))


# In[110]:


multi=pd.read_csv("C:/Users/DELL/Downloads/homeprices.csv")


# In[111]:


multi


# In[116]:


x=multi['bedrooms']


# In[117]:


x


# In[125]:


x.median()


# In[127]:


np.median(x)


# In[136]:


multi.bedrooms = multi.bedrooms.fillna(x.median())


# In[137]:


multi


# In[139]:


multi_reg=linear_model.LinearRegression()
multi_reg.fit(multi[['area','bedrooms','age']],multi.price)


# In[140]:


multi_reg.predict([['3000','3','40']])


# In[141]:


multi_reg.coef_


# In[142]:


multi_reg.predict([['3000','3','1']])


# In[143]:


multi_reg.predict([['3000','3','0']])


# In[144]:


exer_multi=pd.read_csv("C:/Users/DELL/Downloads/hiring.csv")


# In[146]:


exer_multi


# In[151]:


import openpyxl
sample=exer_multi['experience']


# In[152]:


cell=sample['A2']


# In[154]:


exer_multi.experience=exer_multi.experience.fillna("zero")
exer_multi


# In[160]:


import pandas as pd
import numpy as np
from sklearn import linear_model
from word2number import w2n
exer_multi.experience=exer_multi.experience.apply(w2n.word_to_num)


# In[161]:


pip install word2number


# In[162]:


from word2number import w2n


# In[163]:


exer_multi.experience=exer_multi.experience.apply(w2n.word_to_num)


# In[164]:


exer_multi


# In[195]:


x=exer_multi[['test_score(out of 10)']]


# In[196]:


x


# In[197]:


y=x.median()
y


# In[198]:


exer_multi['test_score(out of 10)']=exer_multi['test_score(out of 10)'].fillna(x.median())


# In[199]:


exer_multi


# In[202]:


exer_multi['test_score(out of 10)'] = exer_multi['test_score(out of 10)'].fillna(8.0)


# In[203]:


exer_multi


# In[206]:


reg_multi=linear_model.LinearRegression()
reg_multi.fit(exer_multi[['experience','test_score(out of 10)','interview_score(out of 10)']],exer_multi['salary($)'])


# In[207]:


reg_multi.predict([[12,10,10]])


# In[208]:


pip install ipython


# In[209]:


pip install "ipython[notebook]"


# In[223]:


from sklearn.datasets import load_iris


# In[ ]:





# In[224]:


print (iris.data)


# In[1]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn import linear_model


# In[3]:


fitbit_data = pd.read_csv("C:/Users/DELL/Downloads/data_fatigue.csv")


# In[4]:


fitbit_data


# In[6]:


fitbit_data['age_group'] = pd.cut(fitbit_data['age'], [18, 25, 35, 45, 55, 65], labels=['18-25', '26-35', '36-45', '46-55', '56-65'])


# In[7]:


sleep_data_by_age = fitbit_data.groupby('age_group')


# In[8]:


sleep_data_by_age


# In[9]:


fitbit_data


# In[12]:


age_std = fitbit_data['age'].std()


# In[13]:


age_std


# In[17]:


std_by_hr_group = sleep_data_by_age['hear_rate'].std()


# In[18]:


std_by_hr_group


# In[20]:


mean_hr = fitbit_data['hear_rate'].mean()
std_hr = fitbit_data['hear_rate'].std()


# In[21]:


normal_range = (mean_hr - 2*std_hr, mean_hr + 2*std_hr)


# In[22]:


normal_range


# In[23]:


hr_zscore = (fitbit_data['hear_rate'] - mean_hr) / std_hr


# In[25]:


hr_zscore


# In[38]:


deviant_hr = fitbit_data[hr_zscore < normal_range[0]]
deviant_hr_count = len(deviant_hr)
print('Number of individuals who deviate from the normal heart rate pattern:', deviant_hr_count)


# In[28]:


deviant_hr


# In[42]:


mean_NHR = fitbit_data['norm_heart'].mean()
std_NHR = fitbit_data['norm_heart'].std()


# In[43]:


mean_NHR


# In[44]:


std_NHR


# In[45]:


normal_range_NHR = (mean_NHR - 2*std_NHR, mean_NHR + 2*std_NHR)
normal_range_NHR 


# In[46]:


NHR_zscore = (fitbit_data['norm_heart'] - mean_NHR) / std_NHR
NHR_zscore


# In[47]:


deviant_NHR = fitbit_data[NHR_zscore < normal_range_NHR[0]]
deviant_NHR_count = len(deviant_NHR)
print('Number of individuals who deviate from the normal NHR pattern:', deviant_hr_count)


# In[ ]:





# In[ ]:




