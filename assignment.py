import pandas as pd 
train_df = pd.read_csv('E:\csvdhf5xlsxurlallfiles/train_data.csv')
test_df = pd.read_csv('E:\csvdhf5xlsxurlallfiles/test_data.csv')
label_df = pd.read_csv('E:\csvdhf5xlsxurlallfiles/train_label.csv')
train_test_label_df = pd.concat([train_df, test_df, label_df])
print(train_test_label_df.head())
print(train_test_label_df.info())
#for 1st row
#using countvectorizer
text = [train_test_label_df['text'].iloc[1]]
from sklearn.feature_extraction.text import CountVectorizer
vector = CountVectorizer()
vector.fit(text)
print(vector.vocabulary_)
vector_transform = vector.transform(text)
print(vector_transform.shape)
vector_toarray = vector_transform.toarray()
print(vector_toarray)
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
vectorizer.fit(text)
print(vectorizer.vocabulary_)
print(vectorizer.idf_)
vector1 = vectorizer.transform([text[0]])
print(vector1.shape)
print(vector1.toarray())
from sklearn.feature_extraction.text import HashingVectorizer
hashingvectorizer = HashingVectorizer(n_features=20)
vector2 = hashingvectorizer.transform(text)
print(vector2.shape)
print(vector2.toarray())