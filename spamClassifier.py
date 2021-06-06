import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
dataset=pd.read_csv(r'D:\nlp datatset\SMSSpamCollection',sep='\t',names=["label",'message'])
#print(dataset.head())
corpus=[]
ps =PorterStemmer()
for i in range(0,len(dataset)):
    review=re.sub('[^a-zA-Z]',' ',dataset['message'][i])
    review=review.lower()
    review=review.split()
    review=[ps.stem(word) for word in review if not word in stopwords.words('english')]
    review=' '.join(review)
    corpus.append(review)
print(corpus[1])
cv=CountVectorizer(max_features=500)
x=cv.fit_transform(corpus).toarray()
y=pd.get_dummies(dataset['label'])
y=y.iloc[:,1].values
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)
spamDetectModel=MultinomialNB().fit(x_train,y_train)
yPredict=spamDetectModel.predict(x_test)
confusionMatrix=confusion_matrix(y_test,yPredict)
accuracy=accuracy_score(y_test,yPredict)
print(accuracy)




