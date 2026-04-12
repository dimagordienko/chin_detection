import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

# Создание векторизатора
tfidf_vectorizer = TfidfVectorizer(max_features=1000, 
stop_words='english')

# Примеры документов
documents = [
"Python is a great programming language for text analysis",
"Text classification requires preprocessing and feature extraction",
"Machine learning algorithms can classify text effectively"
]

# Преобразование текстов в TF-IDF векторы
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

# Получение имен признаков (слов)
feature_names = tfidf_vectorizer.get_feature_names_out()

# Вывод TF-IDF значений для первого документа
first_doc_vector = tfidf_matrix[0]
df = pd.DataFrame({'term': feature_names, 
'tfidf': first_doc_vector.toarray()[0]})
df = df.sort_values('tfidf', ascending=False).head(5)
print(df)

