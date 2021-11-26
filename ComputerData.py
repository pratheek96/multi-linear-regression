# Multilinear Regression
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

# loading the data
computer = pd.read_csv("C:\\Users\\Desktop\\Multi Linear Regression\\Computer_Data.csv")

# to get top 6 rows
print(computer.head()) # to get top n rows use cars.head(10)
computer.drop(computer.columns[0],axis=1, inplace=True)
computer = pd.get_dummies(data = computer, columns = [ 'cd', 'multi' , 'premium'])
# Correlation matrix 
print(computer .corr())
input()
# we see there exists High collinearity between input variables especially between
# [Hp & SP] , [VOL,WT] so there exists collinearity problem
 
# Scatter plot between the variables along with histograms
import seaborn as sns
sns.pairplot(computer)
plt.show()


# columns names
print( computer.columns)
input()

# pd.tools.plotting.scatter_matrix(cars); -> also used for plotting all in one graph
                             
# preparing model considering all the variables 
import statsmodels.formula.api as smf # for regression model
         
# Preparing model                  
ml1 = smf.ols('price ~ speed+ hd + ram + screen + ads + trend + cd_no + cd_yes + multi_no + multi_yes + premium_no + premium_yes',data=computer).fit() # regression model
print(ml1) 
# Getting coefficients of variables               
print(ml1.params)

# Summary
print(ml1.summary())
# p-values for WT,VOL are more than 0.05 and also we know that [WT,VOL] has high correlation value 

# preparing model based only on Volume
ml_v=smf.ols('price ~ cd_no ',data = computer ).fit()  
print(ml_v.summary()) # 0.271
# p-value <0.05 .. It is significant 

# Preparing model based only on WT
ml_w=smf.ols('price ~ hd ',data = computer).fit()  
print(ml_w.summary())# 0.268

# Preparing model based only on WT & VOL
ml_wv=smf.ols('price ~ ram  ',data = computer).fit()  
print(ml_wv.summary()) # 0.264

ml_hr=smf.ols('price ~ hd+ram  ',data = computer).fit()  
print(ml_hr.summary())
# Both coefficients p-value became insignificant... 
# So there may be a chance of considering only one among VOL & WT

# Checking whether data has any influential values 
# influence index plots
import statsmodels.api as sm
sm.graphics.influence_plot(ml1)
plt.show()
# index 76 AND 78 is showing high influence so we can exclude that entire row

# Studentized Residuals = Residual/standard deviation of residuals

computer_new=computer.drop(computer.index[[5960,1440,1700,79,4477,3783,19,24,1101,900]],axis=0)

# Preparing model                  
ml_new = smf.ols('price ~ speed+ hd + ram + screen + ads + trend + cd_no + cd_yes + multi_no + multi_yes + premium_no + premium_yes',data = computer_new).fit()    

# Getting coefficients of variables        
print(ml_new.params)

# Summary
print(" Dropping few rows")
print(ml_new.summary()) # 0.806

# Confidence values 99%
print(ml_new.conf_int(0.01)) # 99% confidence level

ml_new = smf.ols('price ~ speed+ hd + ram + screen + ads + trend  + cd_yes + multi_yes + premium_no + premium_yes',data = computer_new).fit()    

# Getting coefficients of variables        
print(ml_new.params)

# Summary
print(" Dropping few rows")
print(ml_new.summary()) # 0.806

# Confidence values 99%
print(ml_new.conf_int(0.01))

# Predicted values of MPG 
mpg_pred = ml_new.predict(computer_new[['speed' , 'hd', 'ram', 'screen', 'ads', 'trend', 'cd_yes', 'multi_yes', 'premium_no', 'premium_yes']])
mpg_pred

print(computer_new.head())
# calculating VIF's values of independent variables
rsq_speed = smf.ols('speed ~hd + ram + screen + ads + trend + cd_yes + multi_yes + premium_no + premium_yes',data=computer_new).fit().rsquared  
vif_speed = 1/(1-rsq_speed) # 16.33

rsq_hd = smf.ols('hd ~speed + ram + screen + ads + trend + cd_yes + multi_yes + premium_no + premium_yes',data=computer_new).fit().rsquared  
vif_hd = 1/(1-rsq_hd) # 564.98

rsq_ram = smf.ols('ram ~hd + speed + screen + ads + trend + cd_yes + multi_yes + premium_no + premium_yes',data=computer_new).fit().rsquared  
vif_ram = 1/(1-rsq_ram) #  564.84

rsq_ads = smf.ols('ads ~hd + ram + screen +speed  + trend + cd_yes + multi_yes + premium_no + premium_yes',data=computer_new).fit().rsquared  
vif_ads = 1/(1-rsq_ads) #  16.

rsq_trend = smf.ols('trend ~hd + ram + screen +speed  + ads + cd_yes + multi_yes + premium_no + premium_yes',data=computer_new).fit().rsquared  
vif_trend = 1/(1-rsq_trend) #  16.35

           # Storing vif values in a data frame
d1 = {'Variables':['speed','hd','ram','ads', 'trend'],'VIF':[vif_speed,vif_hd,vif_ram,vif_ads,vif_trend]}
Vif_frame = pd.DataFrame(d1)  
print(Vif_frame)
# As weight is having higher VIF value, we are not going to include this prediction model

# Added varible plot 
sm.graphics.plot_partregress_grid(ml_new)
plt.show()

# added varible plot for weight is not showing any significance 

# final model
final_ml= smf.ols('price ~ speed+ hd + ram + screen + ads+ trend + cd_yes + multi_yes + premium_no + premium_yes',data = computer_new).fit()
print(final_ml.params)
print("Final summary ")
input()
print(final_ml.summary()) # 0.809
# As we can see that r-squared value has increased from 0.810 to 0.812.

mpg_pred = final_ml.predict(computer_new)



import statsmodels.api as sm
# added variable plot for the final model
sm.graphics.plot_partregress_grid(final_ml)


######  Linearity #########
# Observed values VS Fitted values
plt.scatter(computer_new.price,mpg_pred,c="r");plt.xlabel("observed_values");plt.ylabel("fitted_values")
plt.show()

# Residuals VS Fitted Values 
plt.scatter(mpg_pred,final_ml.resid_pearson,c="r"),plt.axhline(y=0,color='blue');plt.xlabel("fitted_values");plt.ylabel("residuals")
plt.show()

########    Normality plot for residuals ######
# histogram
plt.hist(final_ml.resid_pearson) # Checking the standardized residuals are normally distributed
plt.show()
# QQ plot for residuals 
import pylab          
import scipy.stats as st

# Checking Residuals are normally distributed
st.probplot(final_ml.resid_pearson, dist="norm", plot=pylab)


############ Homoscedasticity #######

# Residuals VS Fitted Values 
plt.scatter(mpg_pred,final_ml.resid_pearson,c="r"),plt.axhline(y=0,color='blue');plt.xlabel("fitted_values");plt.ylabel("residuals")

plt.show()

### Splitting the data into train and test data 

from sklearn.model_selection import train_test_split
computer_train,computer_test  = train_test_split(computer_new,test_size = 0.2) # 20% size

# preparing the model on train data 

model_train = smf.ols("price ~ speed+ hd + ram + screen + ads + trend  + cd_yes + multi_yes + premium_no + premium_yes",data=computer_train).fit()

# train_data prediction
train_pred = model_train.predict(computer_train)
print(train_pred)
# train residual values 
train_resid  = train_pred - computer_train.price
print(train_resid)
# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid*train_resid))
print(train_rmse)
# prediction on test data set 
test_pred = model_train.predict(computer_test)
print(test_pred)
# test residual values 
test_resid  = test_pred - computer_test.price
print(test_resid)
# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid*test_resid))
print(test_rmse)