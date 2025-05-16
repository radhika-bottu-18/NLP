import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import re 
import nltk 
from nltk.corpus import stopwords 
from nltk.stem.porter import PorterStemmer 

#2. Load the dataset
dataset = pd.read_csv(r'D:\DataScienceAndAICourse\May-NLP\NLP\5th,6th  - NLP project\4.CUSTOMERS REVIEW DATASET\Restaurant_Reviews.tsv',delimiter='\t', quoting=3)

#3. Text Cleaning 
corpus = []
for i in range(0,1000):
    review =re.sub('[^a-zA-Z]',' ',dataset['Review'][i])
    review = review.lower().split()
    ps =PorterStemmer()
    review = [ps.stem(word) for word in review if word not in set(stopwords.words('english'))]
    corpus.append(' '.join(review))

#4 TF-IDF Vectorization 
from sklearn.feature_extraction.text import TfidfVectorizer
cv= TfidfVectorizer()
x= cv.fit_transform(corpus).toarray()
y= dataset.iloc[:,1].values

#5Train - test split 
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

# 6. Import Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve

def evaluate_model(model,x_train,x_test,y_train,y_test,model_name):
    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)

    print(f'\n---{model_name} ---')
    print('Acccuracy:',accuracy_score(y_test,y_pred))
    print('Bias (Train Score) ', model.score(x_train,y_train))
    print('Variance SCore (Test Score)',model.score(x_test,y_test))
    
    # Probabilities for ROC
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(x_test)[:, 1]
    else:
         y_proba = model.decision_function(x_test)
         y_proba = (y_proba - y_proba.min()) / (y_proba.max() - y_proba.min())  # scale to [0,1]
    
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = roc_auc_score(y_test,y_proba)
    print('AUC Score:',roc_auc)

    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')

# 8. List of Models
models = [
    (LogisticRegression(), "Logistic Regression"),
    (GaussianNB(), "Naive Bayes"),
    (KNeighborsClassifier(), "KNN"),
    (DecisionTreeClassifier(), "Decision Tree"),
    (RandomForestClassifier(), "Random Forest"),
    (SVC(probability=True), "SVM"),
    (XGBClassifier(use_label_encoder=False, eval_metric='logloss'), "XGBoost"),
    (LGBMClassifier(), "LightGBM")
]

# 9. Evaluate and Plot ROC
plt.figure(figsize=(12, 8))
for model, name in models:
    evaluate_model(model, x_train, x_test, y_train, y_test, name)

plt.plot([0, 1], [0, 1], 'k--')
plt.title("ROC Curve Comparison")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()