from django.shortcuts import render
from django.http import HttpResponse
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as pt 
import seaborn as sb
from django.contrib import messages
import csv,io
from django.contrib.auth.models import User
from home.models import Post
from django.shortcuts import render
from django.contrib.staticfiles.storage import staticfiles_storage
import csv
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor  
# Create your views here.
def home(request):
    return render(request, 'home.html')

def add(request):
    pass



def simpregression(a):
    v = a.columns
    x = a.iloc[:,:-1].values
    y = a.iloc[:,1].values

    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 1/3,random_state = 0)
    from sklearn.linear_model import LinearRegression
    regression = LinearRegression()
    regression.fit(x_train, y_train)
    

    #plotting for train set
    pt.scatter(x_test,y_test,color = 'red')
    pt.plot(x_train,regression.predict(x_train),color = 'blue')
    pt.title(v[0]+' vs'+ v[1])
    pt.xlabel(v[0])
    pt.ylabel(v[1])
    pt.show()
    return 1



def multiregression(b):
    #v=b.drop('State',1)
    b = b.select_dtypes(exclude = ['object'])
    def calculate_vif_(X, thresh=7):
        cols = X.columns
        variables = np.arange(X.shape[1])
        dropped=True
        while dropped:
            dropped=False
            c = X[cols[variables]].values
            vif = [variance_inflation_factor(c, ix) for ix in np.arange(c.shape[1])]
    
            maxloc = vif.index(max(vif))
            if max(vif) > thresh:
               print('dropping \'' + X[cols[variables]].columns[maxloc] + '\' at index: ' + str(maxloc))
               variables = np.delete(variables, maxloc)
               dropped=True
            if len(variables) == 1:
                return X[cols[variables]]
                break
    
        print('Remaining variables:')
        print(X.columns[variables])
        return X[cols[variables]]

    enroll = calculate_vif_(b)

    sb.pairplot(enroll)
    pt.show()
    mesg = ['Thanks for using our website']
    return mesg


def logisticregression(c):
    c = c.select_dtypes(exclude = ['object'])
    data1 = c.copy()
    data1['Admitted']=data1['Admitted'].map({'Yes':1,'No':0})
    data1

    y = data1['Admitted']
    x1 = c['SAT']
    x = sm.add_constant(x1)
    reg_log = sm.Logit(y,x)
    results_log= reg_log.fit()
    def f(x,b0,b1):
        return np.array(np.exp(b0+x*b1)/(1+np.exp(b0+x*b1)))
    f_sorted = np.sort(f(x1,results_log.params[0],results_log.params[1]))
    x_sorted = np.sort(np.array(x1))
    pt.scatter(x1,y,color='C0')
    pt.xlabel('SAT',fontsize = 20)
    pt.ylabel('Admitted', fontsize  =20)
    pt.plot(x_sorted,f_sorted,color='green')
    pt.show()    
    results_log.summary()
    np.set_printoptions(formatter={'float':lambda x:"{0:0.2f}".format(x)})
    results_log.predict()

    np.array(data1['Admitted'])
    results_log.pred_table()

    cm_df=pd.DataFrame(results_log.pred_table())
    cm_df.columns=['prediction 0','prediction 1']
    cm_df=cm_df.rename(index={0:'Actual0',1:'Actual1'})
    cm_df

    cm=np.array(cm_df)
    accuracy_train=(cm[0,0]+cm[1,1])/cm.sum()
    accuracy_train
    return 0



def lo(request):
    v1 = request.POST.get('files')
    if 'downloads' in request.POST:
        v2 = request.POST['downloads']
    c = pd.read_csv(v1)
    f = len(c.columns) 
    if v2 == 'Regression':
        if f<=2:
            simp = simpregression(c)
            return render(request, 'result.html')
        else: 
            multi = multiregression(c)
            return render(request, 'result.html')
    if v2 == 'Classification':
        logic = logisticregression(c)
        return render(request, 'result.html')

    if v2 == 'Clustering':
        return HttpResponse('Clustering')

    return HttpResponse('None')

