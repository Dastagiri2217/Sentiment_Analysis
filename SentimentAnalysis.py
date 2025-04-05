#importing Libraries
import numpy as np
import pandas as pd
dataset = pd.read_csv('Restaurant_Reviews.tsv',delimiter='\t',quoting=3)
dataset
import nltk
import re
nltk.download('stopwords')#is,for these type of words what doesnt add ant importance
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer #for stemming the word like playing to play
cleandata= []
for i in range(0,1000):
    review = re.sub(pattern='[^a-zA-Z]',repl=' ',string = dataset['Review'][i])#replaces words which are not a letter
    review = review.lower()#converting to lower Case
    review_words = review.split(' ') #Splitting to words
    review_words = [word for word in review_words if not word in set(stopwords.words('english'))]
    pstream = PorterStemmer()
    review = [pstream.stem(word) for word in review_words]
    review = ' '.join(review)
    cleandata.append(review)
    cleandata[:1500]
    from sklearn.feature_extraction.text import CountVectorizer
cvect = CountVectorizer(max_features=1500)
x = cvect.fit_transform(cleandata).toarray()
y = dataset.iloc[:,1].values
from sklearn.model_selection import train_test_split #to train the data
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state= 0)
x_train.shape,x_test.shape,y_train.shape,y_test.shape
#importing library naive bayes for text classification
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(x_train,y_train)
y_predict = classifier.predict(x_test)
#calculating accuracy,precicision score
from sklearn.metrics import precision_score,accuracy_score,recall_score
pscore = precision_score(y_test,y_predict)
ascore = accuracy_score(y_test,y_predict)
rscore = recall_score(y_test,y_predict)
#or we can print classification report 
from sklearn.metrics import classification_report
print(classification_report(y_test,y_predict))
from sklearn.metrics import confusion_matrix
cmatrix = confusion_matrix(y_test,y_predict)
from sklearn.metrics import ConfusionMatrixDisplay
graph = ConfusionMatrixDisplay(confusion_matrix=cmatrix)
graph.plot()
#plotting graph to cleary visualise confusion matrix
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(10,6))
hmap = sns.heatmap(cmatrix,annot=True,cmap='YlGnBu',xticklabels=['Negative','Positive'],yticklabels = ['Negative','Positive'])
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.show()
from ssl import ALERT_DESCRIPTION_HANDSHAKE_FAILURE
best_accuracy = 0.0
alpha_val = 0.0
for i in np.arange(0.1,1.1,0.1):
    temp_classifier = MultinomialNB(alpha = i)
    temp_classifier.fit(x_train,y_train)
    temp_ypred = temp_classifier.predict(x_test)
    score = accuracy_score(y_test,temp_ypred)
    print("Accuracy Score for alpha = {} is {}%".format(round(i,1),round(score*100,3)))
    if score > best_accuracy:
        best_accuracy = score
        alpha_val = i
print('-----------------------------------------------------------------------')
print("The Best Accuracy Score is {}% with alpha value  as  {}".format(round(best_accuracy*100,2),round(alpha_val,1)))
classifier  = MultinomialNB(alpha = 0.2)
classifier.fit(x_train,y_train)
def predict_review(sample_review):#predicting sentiment 
    sample_review = re.sub(pattern='[^a-zA-Z]',repl=' ',string = sample_review)
    sample_review = sample_review.lower()
    sample_review_words = sample_review.split(' ')
    sample_review_words = [word for word in sample_review_words if not word in set(stopwords.words('english'))]
    ps = PorterStemmer()
    final_review = [ps.stem(word) for word in sample_review_words]
    final_review = ' '.join(final_review)
    temp = cvect.transform([final_review]).toarray()
    return classifier.predict(temp)
review = "Absolutely loved it! Highly recommend to everyone."
if predict_review(review):
    print("Positive Review")
else:
    print("Negative Review")
review = "The service was excellent and the staff was friendly."
if predict_review(review):
    print("Positive Review")
else:
    print("Negative Review")
review = "Great value for money. Will definitely come back!"
if predict_review(review):
    print("Positive Review")
else:
    print("Negative Review")

review = "Exceeded my expectations. Everything was perfect."
if predict_review(review):
    print("Positive Review")
else:
    print("Negative Review")

review = "Delicious food and cozy atmosphere."
if predict_review(review):
    print("Positive Review")
else:
    print("Negative Review")
print("Script completed")