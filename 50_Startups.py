# Multilinear Regression
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

# loading the data
startups = pd.read_csv("C:\\Users\\Desktop\\Multi Linear Regression\\50_Startups.csv")

# to get top 6 rows
print(startups.head()) # to get top n rows use cars.head(10)
startups_dummy = pd.get_dummies(startups, columns = ['State'])
print(startups_dummy.head())

# Correlation matrix 
print(startups_dummy.corr())
input()

# we see there exists High collinearity between input variables especially between
# # [R&D Spend and profit ] , [Marketing Spend and Profit ] so there exists collinearity problem
 
# Scatter plot between the variables along with histograms
import seaborn as sns
sns.pairplot(startups_dummy)
plt.show()

# columns names
print(startups_dummy.columns)
input()

startups_dummy.rename (columns = {'R&D Spend' : 'RDSpend', 'Marketing Spend' : 'MarketingSpend', 'State_New York' : 'State_New_York'}, inplace = True )
print(startups_dummy.columns)
input()
# pd.tools.plotting.scatter_matrix(cars); -> also used for plotting all in one graph
 #['R&D Spend', 'Administration', 'Marketing Spend', 'State', 'Profit'],                            
# preparing model considering all the variables 
import statsmodels.formula.api as smf # for regression model
         
# Preparing model                  
ml1 = smf.ols('Profit~RDSpend+Administration+MarketingSpend+State_California+State_Florida+State_New_York' ,data=startups_dummy).fit() # regression model
input()
# Getting coefficients of variables               
print(ml1.params)

# Summary
print(ml1.summary())
input()
# p-values for WT,VOL are more than 0.05 and also we know that [WT,VOL] has high correlation value 

# Preparing model                  
ml2 = smf.ols('Profit~RDSpend+Administration+MarketingSpend' ,data=startups_dummy).fit() # regression model
input()
# Getting coefficients of variables               
print(ml2.params)

# Summary
print(ml2.summary())
input()
# p-values for WT,VOL are more than 0.05 and also we know that [WT,VOL] has high correlation value 

# preparing model based only on Volume
ml_v=smf.ols('Profit~RDSpend',data = startups_dummy).fit()  
print(ml_v.summary()) # 0.271
# p-value <0.05 .. It is significant 

# Preparing model based only on WT
ml_w=smf.ols('Profit~Administration',data = startups_dummy).fit()  
print(ml_w.summary()) # 0.268

ml_w=smf.ols('Profit~MarketingSpend',data = startups_dummy).fit()  
print(ml_w.summary())

ml_wv=smf.ols('Profit~Administration+MarketingSpend',data = startups_dummy).fit()  
print(ml_wv.summary()) # 0.264

# Preparing model based only on WT & VOL
ml_wv=smf.ols('Profit~RDSpend+MarketingSpend',data = startups_dummy).fit()  
print(ml_wv.summary()) # 0.264
# Both coefficients p-value became insignificant... 
# So there may be a chance of considering only one among VOL & WT

# Checking whether data has any influential values 
# influence index plots

import statsmodels.api as sm
sm.graphics.influence_plot(ml1)
plt.show()
# index 76 AND 78 is showing high influence so we can exclude that entire row

# Studentized Residuals = Residual/standard deviation of residuals

startups_dummy_new=startups_dummy.drop(startups_dummy.index[[49,48,46]],axis=0)


# Preparing model                  
ml_new = smf.ols('Profit~RDSpend+Administration+MarketingSpend+State_California+State_Florida+State_New_York',data = startups_dummy_new).fit()    

# Getting coefficients of variables        
print(ml_new.params)
input()

# Summary
print(ml_new.summary())# 0.806
input()
# Confidence values 99%
print(ml_new.conf_int(0.01)) # 99% confidence level
input()

# Predicted values of MPG 
mpg_pred = ml_new.predict(startups_dummy_new[['RDSpend','Administration','MarketingSpend','State_California','State_Florida','State_New_York']])
df=pd.DataFrame({'Actual': startups_dummy_new.Profit , 'Predict':mpg_pred})
print(df)
print(startups_dummy_new.head())

# calculating VIF's values of independent variables
rsq_rdspend = smf.ols('RDSpend~Administration+MarketingSpend+State_California+State_Florida+State_New_York',data=startups_dummy_new).fit().rsquared  
vif_rdspend = 1/(1-rsq_rdspend) 
print(vif_rdspend )# 16.33

rsq_admin = smf.ols('Administration~RDSpend+MarketingSpend+State_California+State_Florida+State_New_York',data=startups_dummy_new).fit().rsquared  
vif_admin = 1/(1-rsq_admin) # 564.98
print(vif_admin)

rsq_ms = smf.ols('MarketingSpend~Administration+RDSpend+State_California+State_Florida+State_New_York',data=startups_dummy_new).fit().rsquared  
vif_ms = 1/(1-rsq_ms)#  564.84
print(vif_ms)


# Storing vif values in a data frame
d1 = {'Variables':['RDSpend','Administration','MarketingSpend'],'VIF':[vif_rdspend,vif_admin ,vif_ms]}
Vif_frame = pd.DataFrame(d1)  
print(Vif_frame)
# As weight is having higher VIF value, we are not going to include this prediction model

# Added varible plot 
# sm.graphics.plot_partregress_grid(startups_dummy_new)
# plt.show()

# added varible plot for weight is not showing any significance 

# final model
final_ml= smf.ols('Profit~MarketingSpend+Administration',data = startups_dummy_new).fit()
print(final_ml.params)
print(final_ml.summary()) # 0.809
# As we can see that r-squared value has increased from 0.810 to 0.812.

mpg_pred = final_ml.predict(startups_dummy_new)
df=pd.DataFrame({'Actual': startups_dummy_new.Profit , 'Predict':mpg_pred})
print(df)

import statsmodels.api as sm
# added variable plot for the final model
sm.graphics.plot_partregress_grid(final_ml)

plt.show()
######  Linearity #########
# Observed values VS Fitted values
plt.scatter(startups_dummy_new.Profit,mpg_pred,c="r");plt.xlabel("observed_values");plt.ylabel("fitted_values")

# Residuals VS Fitted Values 
plt.scatter(mpg_pred,final_ml.resid_pearson,c="r"),plt.axhline(y=0,color='blue');plt.xlabel("fitted_values");plt.ylabel("residuals")


########    Normality plot for residuals ######
# histogram
plt.hist(final_ml.resid_pearson) # Checking the standardized residuals are normally distributed

# QQ plot for residuals 
import pylab          
import scipy.stats as st

# Checking Residuals are normally distributed
st.probplot(final_ml.resid_pearson, dist="norm", plot=pylab)


############ Homoscedasticity #######

# Residuals VS Fitted Values 
plt.scatter(mpg_pred,final_ml.resid_pearson,c="r"),plt.axhline(y=0,color='blue');plt.xlabel("fitted_values");plt.ylabel("residuals")



### Splitting the data into train and test data 

from sklearn.model_selection import train_test_split
startups_dummy_new_train ,startups_dummy_new_test  = train_test_split(startups_dummy_new,test_size = 0.2) # 20% size

# preparing the model on train data 

model_train = smf.ols("Profit~RDSpend+MarketingSpend",data=startups_dummy_new_train).fit()
print(model_train.summary())

# train_data prediction
train_pred = model_train.predict(startups_dummy_new_train)
print(train_pred)
# train residual values 
train_resid  = train_pred - startups_dummy_new_train.Profit
print(train_resid)
# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid*train_resid))
print(train_rmse)
# prediction on test data set 
test_pred = model_train.predict(startups_dummy_new_test)
print(test_pred)
# test residual values 
test_resid  = test_pred - startups_dummy_new_test.Profit
print(test_resid)
# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid*test_resid))
print(test_rmse )



import sklearn 
from sklearn.model_selection import train_test_split
X = startups_dummy.iloc[:, [0,1,2,4,5,6]].values 
y = startups_dummy.iloc[:, 3].values  
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=0 )
from sklearn.linear_model import LinearRegression 
regressor = LinearRegression()
regressor.fit(X_train, y_train)

print(" Co-eff: ", regressor.coef_)
regressor.score(X_train, y_train)

y_pred = regressor.predict(X_test )
print("Regressor: ", regressor )

print(" Y pred : ", y_pred)

df=pd.DataFrame({'Actual': y_test , 'Predict':y_pred})
print(df)

X = startups_dummy.iloc[:, [0,2]].values 
y = startups_dummy.iloc[:, 3].values  
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=0 )
from sklearn.linear_model import LinearRegression 
regressor = LinearRegression()
regressor.fit(X_train, y_train)

print(" Co-eff: ", regressor.coef_)
regressor.score(X_train, y_train)

y_pred = regressor.predict(X_test )
print("Regressor: ", regressor )

print(" Y pred : ", y_pred)

df=pd.DataFrame({'Actual': y_test , 'Predict':y_pred})
print(df)