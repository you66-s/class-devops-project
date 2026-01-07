# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer

import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support,roc_curve, auc, accuracy_score

# Model Evaluation
def model_evaluation(yt, yp):
    results = {}
    results['accuracy'] = accuracy_score(yt, yp)

    precision, recall, fscore, _ = precision_recall_fscore_support(yt, yp, average = 'weighted')
    results['precision'] = precision
    results['recall'] = recall
    results['fscore'] = fscore

    metrics = list(results.keys())
    values = list(results.values())

    ax = sns.barplot(x = metrics, y = values, palette = 'viridis')
    plt.title('Model Evaluation Metrics')
    plt.ylim(0,1)
    plt.ylabel('Value')

    for i, v in enumerate(values):
        plt.text(i, v/2, f'{v: 0.2f}', ha = 'center', va = 'center', color = 'white',
                fontsize = 12)
    plt.show()

# Classification Report
def class_report(yt, yp):
    cr = pd.DataFrame(classification_report(yt, yp, output_dict = True))
    return cr.T.style.background_gradient(cmap = 'Blues', axis = 0)

# Confusion Matrix
def conf_matrix(yt, yp):
    cm = confusion_matrix(yt, yp)
    sns.heatmap(cm, annot = True, linecolor = 'black', fmt = '0.2f', cmap = 'Blues',
               linewidths = 0.01)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Class')
    plt.ylabel('Actual Class')
    plt.show()

def plot_roc(yt, yp):
    fpr, tpr, thr = roc_curve(yt, yp)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize = (6,6))
    plt.plot(fpr, tpr, color = 'darkorange', lw=2,
            label = f"ROC_CURVE (Area = {roc_auc : 0.3f})")
    plt.plot([0.0, 1.0], [0.0, 1.0], lw=2, color = 'navy', linestyle = '--')
    plt.xlim([-0.01, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc = 'lower right')
    plt.show()
    df = pd.read_csv(r'/kaggle/input/email-spam-classification-dataset/combined_data.csv')
df.head().style.background_gradient(cmap = 'Blues', axis = 0)
# Defining the dimension
print(f'Dimension of the Dataset: {df.shape}\n')

# Discriptive Statistics
print(f'Discriptive Statistics: {df.describe()}\n')

# Data Info
print(df.info())
# Finding the null vaues
df.isnull().sum()
df.head()
# Distribution of spam and non-spam email
plt.figure(figsize = (5,5))

ax = sns.countplot(x = df['label'], data = df, hue = 'label', palette = 'viridis')
plt.title('Distribution of the spam and non-spam')
plt.xlabel('is_spam')

for i, v in enumerate(list(df['label'].value_counts())):
  plt.text(i, v/2 , f'{v}', ha = 'center', va = 'center', color = 'white', fontsize = 12)
plt.show()
# Finding the most common used words
from collections import Counter
from wordcloud import WordCloud

# Functions to get most common words
def get_common_words(text, n = 20):
  words = ' '.join(text).split()
  common_words = Counter(words)
  return common_words

# Seperate spam and non-spam

score_0 = df[df['label'] == 0]['text']
score_1 = df[df['label'] == 1]['text']

# Get most common words

score0_common_words = get_common_words(score_0)
score1_common_words = get_common_words(score_1)

# Generate word clouds
score0_word_cloud = WordCloud(width = 800, height = 400).generate(' '.join(score0_common_words))
score1_word_cloud = WordCloud(width = 800, height = 400).generate(' '.join(score0_common_words))

# Plot word clouds
score_word_clouds = [score0_word_cloud, score1_word_cloud]
plt.figure(figsize = (20,20))

for i in range(2):
  plt.subplot(1, 2, i+1)
  plt.imshow(score_word_clouds[i], interpolation = 'bilinear')
  plt.axis('off')
  plt.title(f'is_spam = {i}')
plt.show()
train_text, test_text, train_labels, test_labels = train_test_split(df['text'], df['label'], test_size = 0.30, random_state = 42)

print(train_text.shape)
print(test_text.shape)

print(train_labels.shape)
print(test_labels.shape)
# Converting text to a number vector
tfd_idk = TfidfVectorizer()
train_text_tfdidk = tfd_idk.fit_transform(train_text)
test_text_tfdidk = tfd_idk.transform(test_text)
from sklearn.linear_model  import LogisticRegression

params_dist = {
    'penalty': ['l1', 'l2'],
    'C' : [0, 0.001, 0.2, 0.4, 0.6, 0.8, 1.0, 10, 100, 1000]
}

log_reg = LogisticRegression(max_iter = 1000, random_state = 42)

log_reg_random_search = RandomizedSearchCV(estimator = log_reg, param_distributions = params_dist, cv=5,
                                           scoring = 'accuracy', n_jobs = -1, n_iter = 10)
log_reg_random_search.fit(train_text_tfdidk, train_labels)

print(f'Best parameters : {log_reg_random_search.best_params_}')
print(f'Best Score : {log_reg_random_search.best_score_}')
y_pred = log_reg_random_search.predict(test_text_tfdidk)
class_report(test_labels, y_pred)
model_evaluation(test_labels, y_pred)
conf_matrix(test_labels, y_pred)
plot_roc(test_labels, log_reg_random_search.predict_proba(test_text_tfdidk)[:,1])
from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB()

params_dist = {
    'alpha' : [0.01, 0.1, 1, 10, 100, 1000]
}

mnb_random_search = RandomizedSearchCV(estimator = mnb, param_distributions = params_dist, cv=5,
                                           scoring = 'accuracy', n_jobs = -1, n_iter = 10)
mnb_random_search.fit(train_text_tfdidk, train_labels)

print(f'Best parameters : {mnb_random_search.best_params_}')
print(f'Best Score : {mnb_random_search.best_score_}')
y_pred = mnb_random_search.predict(test_text_tfdidk)
class_report(test_labels, y_pred)
model_evaluation(test_labels, y_pred)
conf_matrix(test_labels, y_pred)
plot_roc(test_labels, mnb_random_search.predict_proba(test_text_tfdidk)[:,1])
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(random_state = 42)

params_dist = {
    'n_estimators' : [10 * n for n in range(1,21)],
    'max_depth' : [2 * n for n in range(1,21)],
    'min_samples_leaf' : [n for n in range(1,11)],
    'min_samples_split' : [n for n in range(1,11)]
}

rf_random_search = RandomizedSearchCV(estimator = rf, param_distributions = params_dist,
                                  cv=5, n_jobs = -1, n_iter = 10, scoring = 'accuracy')
rf_random_search.fit(train_text_tfdidk, train_labels)

print(f'Best Parameters : {rf_random_search.best_params_}')
print(f"Best Score: {rf_random_search.best_score_}")
y_pred = rf_random_search.predict(test_text_tfdidk)
class_report(test_labels, y_pred)
import joblib

joblib.dump(tfd_idk, "tfidf.pkl")          # vectoriseur
joblib.dump(log_reg, "spam_model.pkl") # modèle entraîné
