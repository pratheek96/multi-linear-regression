# Multilinear Regression
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

# loading the data
Toyotta = pd.read_csv("C:\\Users\\Desktop\\Multi Linear Regression\\ToyotaCorolla.csv", encoding = 'latin1')

# to get top 6 rows
print(Toyotta.head()) # to get top n rows use cars.head(10)
Toyota = Toyotta.iloc[:, [2,3,6,8,12,13,15,16,17]]
# Correlation matrix 
print(Toyota.corr())
input()

# we see there exists High collinearity between input variables especially between
# # [R&D Spend and profit ] , [Marketing Spend and Profit ] so there exists collinearity problem
 
# Scatter plot between the variables along with histograms
import seaborn as sns
sns.pairplot(Toyota)
plt.show()

# columns names

input()
# pd.tools.plotting.scatter_matrix(cars); -> also used for plotting all in one graph
 #['R&D Spend', 'Administration', 'Marketing Spend', 'State', 'Profit'],                            
# preparing model considering all the variables 
import statsmodels.formula.api as smf # for regression model
         
# Preparing model                  
ml1 = smf.ols('Price~Age_08_04 + KM + HP + cc + Doors + Gears + Quarterly_Tax + Weight ' ,data=Toyota).fit() # regression model
input()
# Getting coefficients of variables               
print(ml1.params)

# Summary
print(ml1.summary())
input()
# p-values for WT,VOL are more than 0.05 and also we know that [WT,VOL] has high correlation value 

# Preparing model                  
# preparing model based only on Volume
ml_v=smf.ols('Price~Doors',data = Toyota).fit()  
print(ml_v.summary()) # 0.271
# p-value <0.05 .. It is significant 

# Preparing model based only on WT
ml_w=smf.ols('Price~cc',data = Toyota).fit()  
print(ml_w.summary()) # 0.268

# Preparing model based only on WT & VOL
ml_wv=smf.ols('Price~Doors+cc',data = Toyota).fit()  
print(ml_wv.summary())
# # Checking whether data has any influential values 
# # influence index plots

import statsmodels.api as sm
sm.graphics.influence_plot(ml1)
plt.show()
# # index 76 AND 78 is showing high influence so we can exclude that entire row

Toyota_new=Toyota.drop(Toyota.index[[80,960,221,991,956]],axis=0)

# Preparing model                  
ml_new = smf.ols('Price~Age_08_04 + KM + HP + cc + Doors + Gears + Quarterly_Tax + Weight ',data = Toyota_new).fit() 

# Getting coefficients of variables        
ml_new.params

# Summary
print(" Dropping few rows")
print(ml_new.summary()) # 0.806

# Confidence values 99%
print(ml_new.conf_int(0.01))


# Predicted values of MPG 
mpg_pred = ml_new.predict(Toyota_new[['Age_08_04', 'KM' , 'HP' , 'cc','Doors' , 'Gears' ,'Quarterly_Tax','Weight']])
mpg_pred

print(Toyota_new.head())
# calculating VIF's values of independent variables
rsq_age = smf.ols('Age_08_04~KM + HP + cc + Doors + Gears + Quarterly_Tax + Weight',data=Toyota_new).fit().rsquared  
vif_age = 1/(1-rsq_age) # 16.33

rsq_km = smf.ols(' KM ~ Age_08_04 + HP + cc + Doors + Gears + Quarterly_Tax + Weight',data=Toyota_new).fit().rsquared  
vif_km= 1/(1-rsq_km) # 564.98

rsq_hp = smf.ols(' HP~ Age_08_04 + KM + cc + Doors + Gears + Quarterly_Tax + Weight',data=Toyota_new).fit().rsquared  
vif_hp = 1/(1-rsq_hp) #  564.84

rsq_cc = smf.ols(' cc~ Age_08_04 + KM + HP + Doors + Gears + Quarterly_Tax + Weight',data=Toyota_new).fit().rsquared  
vif_cc = 1/(1-rsq_cc) #  16.35

rsq_doors = smf.ols(' Doors~ cc+ Age_08_04 + KM + HP + Gears + Quarterly_Tax + Weight',data=Toyota_new).fit().rsquared  
vif_doors = 1/(1-rsq_doors) #  16.35

rsq_gears = smf.ols('Gears ~ cc+  Age_08_04 + KM + HP + Doors + Quarterly_Tax + Weight',data=Toyota_new).fit().rsquared  
vif_gears = 1/(1-rsq_gears) #  16.35


rsq_qt = smf.ols(' Quarterly_Tax ~ cc+ Age_08_04 + KM + HP + Doors + Gears + Weight',data=Toyota_new).fit().rsquared  
vif_qt = 1/(1-rsq_qt) #  16.35


rsq_wt = smf.ols('Weight ~ cc+ Age_08_04 + KM + HP + Doors + Gears + Quarterly_Tax  ',data=Toyota_new).fit().rsquared  
vif_wt = 1/(1-rsq_wt) #  16.35

           # Storing vif values in a data frame
d1 = {'Variables':['Age_08_04', 'KM' , 'HP' , 'cc','Doors' , 'Gears' ,'Quarterly_Tax','Weight'],'VIF':[vif_age,vif_km,vif_hp,vif_cc,vif_doors,vif_gears,vif_qt,vif_wt]}
Vif_frame = pd.DataFrame(d1)  
print(Vif_frame)
# As weight is having higher VIF value, we are not going to include this prediction model

# Added varible plot 
sm.graphics.plot_partregress_grid(ml_new)
plt.show()


# final model
final_ml= smf.ols('Price~Age_08_04 + KM + HP + Doors + Gears + Quarterly_Tax',data = Toyota_new).fit()
print(final_ml.params)
print("Final summary ")
input()
print(final_ml.summary()) # 0.809
# As we can see that r-squared value has increased from 0.810 to 0.812.

mpg_pred = final_ml.predict(Toyota_new)

import statsmodels.api as sm
# added variable plot for the final model
sm.graphics.plot_partregress_grid(final_ml)
plt.show()

######  Linearity #########
# Observed values VS Fitted values
plt.scatter(Toyota_new.Price,mpg_pred,c="r");plt.xlabel("observed_values");plt.ylabel("fitted_values")

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
plt.show()



### Splitting the data into train and test data 

from sklearn.model_selection import train_test_split
cars_train,cars_test  = train_test_split(Toyota_new,test_size = 0.2) # 20% size

# preparing the model on train data 

model_train = smf.ols("Price~Age_08_04 + KM + HP + Doors + Gears + Quarterly_Tax",data=cars_train).fit()

# train_data prediction
train_pred = model_train.predict(cars_train)
print(train_pred)

# train residual values 
train_resid  = train_pred - cars_train.Price
print(train_resid)
# RMSE value for train data 
input()
train_rmse = np.sqrt(np.mean(train_resid*train_resid))
print(train_rmse)
# prediction on test data set 
input()
test_pred = model_train.predict(cars_test)
print(test_pred)
# test residual values 
input()
test_resid  = test_pred - cars_test.Price
print(test_resid)
# RMSE value for test data 
input()
test_rmse = np.sqrt(np.mean(test_resid*test_resid))
print(test_rmse)