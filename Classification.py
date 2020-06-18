import re
from nltk.stem.porter import PorterStemmer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

'''Creating Bag of Words model'''
corpus = []
dataset = open("LabelledData.txt", "r").readlines()
print(len(dataset)) #1483

'''Splitting each line into question and its class'''
for i in range(0, len(dataset)):
    dataset[i] = dataset[i].split(" ,,, ")
        
'''Preprocessing of text
    1. Cleaning of text that contains special and unwanted characters using re(regular expression) library
    2. Lower casing the text
    3. Reducing the word dimensions using porter stemmer(maps words to its root words)
    4. Appending the words to the corpus.
    '''    
for i in range(0, len(dataset)):
    question = dataset[i][0]
    question = re.sub('[^a-zA-Z]', ' ', dataset[i][0])
    question = question.lower()
    question = question.split()
    ps = PorterStemmer()
    question = [ps.stem(word) for word in question]
    question = ' '.join(question)
    corpus.append(question)
    
'''Feature extraction: 
    Extracting the features from the corpus using CountVectorizer'''    
cv = CountVectorizer(max_features = 1500)

'''Fitting the count vectorizer to the questions'''
X = cv.fit_transform(corpus).toarray()

'''Features that are extracted from the corpus'''
print(cv.get_feature_names())

'''Preprocessing on the classes'''
y = []
for i in range(0, len(dataset)):
    question_class = re.sub('[^a-zA-Z]', '', dataset[i][1])
    y.append(question_class)


print(len(X))
print(len(y))
print(list(set(y)))

'''Test Train split of the dataset using train_test_split with
    20 percent test size
    80 percent train size'''
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


'''Importing and fitting the Logistic Regression classifier to X_train and y_train'''
from sklearn.linear_model import LogisticRegression
classifier_lg = LogisticRegression()
classifier_lg.fit(X_train, y_train)

'''Prediction on the test dataset using Logistic Regression'''
y_pred_lg = classifier_lg.predict(X_test)

'''Confusion matrix, classification report and accuracy score
    on test dataset using Logistic Regression'''
cm_lg= confusion_matrix(y_test, y_pred_lg)
print(cm_lg)
print(classification_report(y_test, y_pred_lg, labels=['affirmation', 'what', 'when', 'unknown', 'who']))
print(accuracy_score(y_test, y_pred_lg))



'''Importing and fitting the K-Neighbors classifier to X_train and y_train'''
from sklearn.neighbors import KNeighborsClassifier
classifier_knn = KNeighborsClassifier()
classifier_knn.fit(X_train, y_train)

'''Prediction on the test dataset using K-Neighbors classifier'''
y_pred_knn = classifier_knn.predict(X_test)

'''Confusion matrix, classification report and accuracy score
    on test dataset using K-Neighbors classifier'''
cm_knn= confusion_matrix(y_test, y_pred_knn)
print(cm_knn)
print(classification_report(y_test, y_pred_knn, labels=['affirmation', 'what', 'when', 'unknown', 'who']))
print(accuracy_score(y_test, y_pred_knn))



'''Importing and fitting the Decision Tree classifier to X_train and y_train'''
from sklearn.tree import DecisionTreeClassifier
classifier_dtc = DecisionTreeClassifier()
classifier_dtc.fit(X_train, y_train)

'''Prediction on the test dataset using Decision Tree Classifier'''
y_pred_dtc = classifier_dtc.predict(X_test)

'''Confusion matrix, classification report and accuracy score
    on test dataset using Decision Tree Classifier'''
cm_dtc= confusion_matrix(y_test, y_pred_dtc)
print(cm_dtc)
print(classification_report(y_test, y_pred_dtc, labels=['affirmation', 'what', 'when', 'unknown', 'who']))
print(accuracy_score(y_test, y_pred_dtc))





