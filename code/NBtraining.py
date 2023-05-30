#import needed packages
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import pickle
import pandas as pd

#load training data into file
df = pd.read_csv("/Users/arthurwu/Desktop/training_dataset_NB.csv")

#vectorize the text dataset we are using
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['text'])
Y = df['ruling'].map({'mute':1, 'nomute':0})

#split dataset into training and testing dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

#initialize the Naive Bayes algorithm
classifier = MultinomialNB()
classifier.fit(X_train, Y_train)

#create the accuracy score
Y_pred = classifier.predict(X_test)
acc = accuracy_score(Y_test, Y_pred)

#if the accuracy is above 0.8, save the machine learning algorithm.
#Otherwise, send a warning and do not save it.
if acc > 0.8:
    with open('NBclassifier.pickle', 'wb') as f:
        pickle.dump(classifier, f)
    with open('vectorizer.pickle', 'wb') as f:
        pickle.dump(vectorizer, f)
else:
    print(f"The accuracy was:{acc}. You should consider retraining the model.")


