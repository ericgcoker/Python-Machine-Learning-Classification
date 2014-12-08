import numpy as np
from sklearn.metrics import classification_report
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from matplotlib import pylab
from sklearn.linear_model import LogisticRegression
import os
import sqlite3 as lite
import pandas as pd
from pandas.io import sql
from sklearn.feature_extraction import DictVectorizer as DV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.metrics import roc_auc_score as AUC
from sklearn.metrics import confusion_matrix
import statsmodels.api as sm
import pylab as pl
import matplotlab as plt

dir=('C:\\educational\\')
#try:
#	con=lite.connect(dir + 'card_data_v1.db')

#	cur=con.cursor()
#	cur.execute('SELECT SQLITE_VERSION()')

#	data=cur.fetchone()

#	print "SQLite version: %s" % data

#except lite.Error, e:

#	print "Error %s:" % e.args[0]
#	sys.exit(1)
	
#<sqlite3.Cursor object at 0x024D9860>
#SQLite version: 3.6.21


cnx=lite.connect(dir + 'card_data_v1.db')

p=sql.read_sql('select * from bt_data', cnx)
#p.head()
#  index  dispute  received_discount  transaction_amount        industry  \
#0      0        0                  0                 685         Service   
#1      1        1                  0                 217        Software   
#2      2        0                  0                  43  Physical_goods   
#3      3        1                  0                 589        Software   
#4      4        0                  0                  60        Software   

#    card_type  
#0      Paypal  
#1        Visa  
#2  MasterCard  
#3      Paypal  
#4  MasterCard  
pd.crosstab(p.industry, p.card_type, margins=True)
#card_type       MasterCard  Paypal  Visa  visa   All
#industry                                            
#Physical_goods         126      93   317     3   539
#Service                149     175   418    10   752
#Software               297     292   734    16  1339
#All                    573     561  1475    29  2638

print q.describe()
 #           index      dispute  received_discount  transaction_amount  \
#count  2630.000000  2630.000000        2630.000000         2630.000000   
#mean   1317.451711     0.344867           0.090494          383.554373   
#std     761.022517     0.475415           0.286943          551.239668   
#min       0.000000     0.000000           0.000000         -500.000000   
#25%     659.250000     0.000000           0.000000           83.000000   
#50%    1316.500000     0.000000           0.000000          168.000000   
#75%    1975.750000     1.000000           0.000000          486.000000   
#max    2637.000000     1.000000           1.000000         2999.000000   
#
#      industry_code  card_type_code  
#count    2630.000000     2630.000000  
#mean        1.304183        1.352091  
#std         0.788529        0.814467  
#min         0.000000        0.000000  
#25%         1.000000        1.000000  
#50%         2.000000        2.000000  
#75%         2.000000        2.000000  
#max         2.000000        2.000000  

pd.crosstab(p.industry, p.card_type).plot(kind="bar")
l=p.card_type
c=[item.lower() for item in l]
p.card_type=c
q=p.dropna(axis=0)
#r=p.fillna('unknown', axis=0)

q['industry_code']=pd.Categorical.from_array(q.industry).labels
q['card_type_code']=pd.Categorical.from_array(q.card_type).labels

d_cats=pd.get_dummies(q['industry_code'], prefix='industry')
d_cats2=pd.get_dummies(q['card_type_code'], prefix='card_type')
d_cats3=pd.get_dummies(q['received_discount'], prefix='received_discount')
columns=['dispute', 'transaction_amount']
data=q[columns].join(d_cats.ix[:, 'industry_1':])
data2=data.join(d_cats2.ix[:, 'card_type_1':])
data3=data2.join(d_cats3.ix[:, 'received_discount_1':])
data3['intercept'] = 1.0

cols=['received_discount_1', 'transaction_amount', 'industry_1', 'industry_2', 'card_type_1', 'card_type_2']
logit = sm.Logit(data3['dispute'], data3[cols])
result = logit.fit()
y_pred = result.predict(data3[cols])
print result.summary()
print result.conf_int()
print np.exp(result.params)
                           Logit Regression Results                           
# ==============================================================================
# Dep. Variable:                dispute   No. Observations:                 2630
# Model:                          Logit   Df Residuals:                     2624
# Method:                           MLE   Df Model:                            5
# Date:                Sun, 09 Nov 2014   Pseudo R-squ.:                  0.1545
# Time:                        22:08:59   Log-Likelihood:                -1432.5
# converged:                       True   LL-Null:                       -1694.3
                                        # LLR p-value:                6.437e-111
# =======================================================================================
                          # coef    std err          z      P>|z|      [95.0% Conf. Int.]
# ---------------------------------------------------------------------------------------
# received_discount_1    -1.7496      0.208     -8.428      0.000        -2.156    -1.343
# transaction_amount      0.0017      0.000     14.735      0.000         0.001     0.002
# industry_1             -2.1442      0.133    -16.120      0.000        -2.405    -1.884
# industry_2             -0.4967      0.088     -5.628      0.000        -0.670    -0.324
# card_type_1            -0.5131      0.122     -4.192      0.000        -0.753    -0.273
# card_type_2            -0.0963      0.088     -1.097      0.273        -0.268     0.076
# ====================================================================================

# received_discount_1 -2.156401 -1.342700
# transaction_amount   0.001468  0.001918
# industry_1          -2.404948 -1.883519
# industry_2          -0.669691 -0.323719
# card_type_1         -0.753008 -0.273168
# card_type_2         -0.268305  0.075719

# received_discount_1    0.173852
# transaction_amount     1.001694
# industry_1             0.117158
# industry_2             0.608532
# card_type_1            0.598644
# card_type_2            0.908198
# dtype: float64

#use scikit learn and see if other classification gives more precision and recall than LR
train, test=train_test_split(data3, test_size = .8, random_state=0)
training=pd.DataFrame(train, columns=['dispute', 'transaction_amount', 'industry_1', 'industry_2', 'card_type_1', 'card_type_2', 'received_discount', 'intercept'])
testing=pd.DataFrame(test, columns=['dispute', 'transaction_amount', 'industry_1', 'industry_2', 'card_type_1', 'card_type_2', 'received_discount', 'intercept'])

numeric_cols =['transaction_amount']
x_num_tr=training[numeric_cols].as_matrix()
x_num_te=testing[numeric_cols].as_matrix()

y_train=training.dispute
y_test=testing.dispute

categ_tr=training.drop( numeric_cols + ['intercept', 'dispute'], axis=1)
categ_te=testing.drop( numeric_cols + ['intercept', 'dispute'], axis=1)

x_categ_tr=categ_tr.T.to_dict().values()
x_categ_te=categ_te.T.to_dict().values()

vectorizer=DV(sparse=False)
vec_x_cat_train=vectorizer.fit_transform(x_categ_tr)
vec_x_cat_test=vectorizer.fit_transform(x_categ_te)

x_train=np.hstack((x_num_tr, vec_x_cat_train))
x_test=np.hstack((x_num_te, vec_x_cat_test))

# fit on training data
lr_clf = LogisticRegression()
lr_clf = lr_clf.fit(x_train, y_train)
lr_predicted = lr_clf.predict(x_test)
#print coefficients
print lr_clf.coef_, lr_clf.intercept_
# [[ 0.00227402  0.45948647  0.91093951 -1.03693812  0.89880819 -1.25285417]] [-2.27181555]
# print classification report
target_names = ['no_dispute','dispute']
print 'Logistic Regression Classification Report'
print (classification_report(y_test, lr_predicted, target_names = target_names))
             # precision    recall  f1-score   support

 # no_dispute       0.76      0.92      0.83      1377
    # dispute       0.74      0.45      0.56       727

# avg / total       0.75      0.76      0.74      2104
# support vector machine classifier
from sklearn.linear_model import SGDClassifier
svm = SGDClassifier()
svm = svm.fit(x_train, y_train)
svm_predicted = svm.predict(x_test)
print 'Support Vector Machine Classification Report:'
print (classification_report(y_test, svm_predicted, target_names = target_names))
             # precision    recall  f1-score   support

 # no_dispute       0.65      1.00      0.79      1377
    # dispute       0.17      0.00      0.00       727

# avg / total       0.49      0.65      0.52      2104


# naive bayes classifier
from sklearn.naive_bayes import MultinomialNB
nb_clf = MultinomialNB()
nb_clf = nb_clf.fit(x_train, y_train)
nb_predicted = nb_clf.predict(x_test)
print 'Naive Bayes Classification Report:'
print (classification_report(y_test, nb_predicted, target_names = target_names))
             # precision    recall  f1-score   support

 # no_dispute       0.75      0.86      0.80      1377
    # dispute       0.63      0.46      0.53       727

# avg / total       0.71      0.72      0.71      2104


# Compute confusion matrix
cm = confusion_matrix(y_test, lr_predicted)
print(cm)
#[[1264  113]
 #[ 401  326]]






	
