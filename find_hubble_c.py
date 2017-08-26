import matplotlib.pyplot as plt 
import numpy as np 	
from sklearn.linear_model import LinearRegression
import pandas

plt.style.use('seaborn')

# http://leda.univ-lyon1.fr/leda/fullsql.html
# SQL QUERY: SELECT objname, modbest, vgsr WHERE modbest IS NOT NULL
df = pandas.read_csv('fullsqlmean (2).cgi', delimiter=',')
df.columns = ['a','b','c','d']

# d(parsec) = 10^(1+0.5*modbest)
# d(Megaparsec) = d(parsec)/(10^6)
X = (10**(1+df.b.values/5)).reshape(-1,1)/(10**6)
y = df.c.values.reshape(-1,1)

#Line of best fit
reg = LinearRegression().fit(X,y)

######
######

plt.subplot(2,1,1)
plt.scatter(X,y,s=0.5, color='red')
plt.annotate(('$H_{0}=%s$'%reg.coef_[0][0]),xy=(2000,200000),xytext=(2000,200000)) 
plt.plot(X,reg.predict(X),linewidth=0.3)
plt.title('$H_{0}$'),plt.ylabel('Velocity') 

print(reg.coef_[0][0],'hubble constant')
plt.subplot(2,1,2)
plt.scatter(X,y,s=0.01, color='red')
plt.plot(X,reg.predict(X),linewidth=0.3)
plt.xlim(0,500),plt.ylim(0,40000)
plt.xlabel('Mpc'),plt.ylabel('Velocity')
plt.show()

t = 3.08e19/(reg.coef_*365.25*24*60*60)

print(t[0][0],'yrs','\n',t[0][0]/(10**9),'billion yrs') # age_of_universe
