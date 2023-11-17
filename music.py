import numpy as np # linear algebra
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import csr_matrix
songs1= pd.read_csv('tcc_ceds_music.csv')
songs = songs1.head(10000)
songs.head()
songs.isnull().sum()
songs.shape
songs.columns
unique_songs = songs['track_name'].unique().shape[0]
print(f"There are {unique_songs} unique songs in the dataset")
unique_artist=songs['artist'].unique().shape[0]
print(f"There are {unique_artist} unique artist in the dataset")
release_songs=songs['release_date'].value_counts()
release_songs
count_genre=songs["genre"].value_counts()
count_genre
count_topic=songs['topic'].value_counts()
count_topic
songs['release_date'].plot.line()
df = songs[songs.genre.isin(songs.genre.value_counts().index)]

sns.boxplot(
    x='genre',
    y='release_date',
    data=df
)
songs_cpy = songs[['artist_name', 'track_name', 'genre', 'lyrics', 'topic']]
songs_cpy.columns
songs_topic=songs[['sadness','violence','world/life',
'obscene',
'music',
'night/time',
'romantic',
'feelings' ]]
songs_topic.columns
import seaborn as sns
sns.set_theme(style="ticks")
sns.pairplot(songs_topic)
import ast
songs_cpy['tags'] =   songs_cpy['topic']+" "+songs_cpy['genre'] + " "+songs_cpy['lyrics'] +" "+ songs_cpy['artist_name']
new = songs_cpy.drop(columns=['topic','genre','lyrics','artist_name'])
songs_cpy['tags'].head(2)
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000, stop_words='english')
vector = cv.fit_transform(new['tags']).toarray()
vector.shape
from sklearn.metrics.pairwise import cosine_similarity
similarity=cosine_similarity(vector)
similarity
def recommend_song(song):
    index = new[new['track_name'] == song].index[0]
    distances = sorted(list(enumerate(similarity[index])),reverse=True,key = lambda x: x[1])
    for i in distances[1:6]:
        print(new.iloc[i[0]].track_name)
     recommend_song('smalltown boy') 

