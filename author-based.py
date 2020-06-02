import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

#loading the data in books variable using pandas
books = pd.read_csv('dataset/books.csv', encoding = "ISO-8859-1")


#TfIdf 
tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
tfidf_matrix = tf.fit_transform(books['authors'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

#storing column in 
titles = books['title']
g_id=books['goodreads_book_id']
url=books['image_url']

indices = pd.Series(books.index, index=books['title'])

# Function that get book recommendations based on the cosine similarity score of book authors
def authors_recommendations(title):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:21]
    book_indices = [i[0] for i in sim_scores]
    #for i in book_indices:
      #  book_indices=i+1
    
    mix={"title":[titles.iloc[book_indices],g_id.iloc[book_indices],url.iloc[book_indices]]}
    # print("Titles are:",titles.iloc[book_indices])
    # print("Good_read books Id is",g_id.iloc[book_indices])
    # print("URL is",url.iloc[book_indices])
    print(mix)
        
    
    #print(books[books["goodreads_book_id","title"]]==titles.iloc[book_indices])
    #return mix

authors_recommendations('The Hobbit')

