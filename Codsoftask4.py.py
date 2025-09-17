import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

data = {
    'title': [
        'Inception', 'The Matrix', 'Interstellar',
        'The Notebook', 'Titanic', 'Avengers: Endgame'
    ],
    'genre': [
        'Sci-Fi Action', 'Sci-Fi Action', 'Sci-Fi Drama',
        'Romance Drama', 'Romance Drama', 'Action Superhero'
    ],
    'description': [
        'A mind-bending thriller about dreams and time travel.',
        'A hacker discovers the truth about reality and rebellion.',
        'Explorers travel through space and time to save humanity.',
        'A romantic story of love and memory through the years.',
        'A tragic love story set aboard the ill-fated Titanic.',
        'Superheroes unite to battle a cosmic threat and save the universe.'
    ]
}

df = pd.DataFrame(data)

df['combined'] = df['genre'] + ' ' + df['description']

vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['combined'])

cosine_sim = cosine_similarity(tfidf_matrix)

def recommend(title, top_n=3):
    if title not in df['title'].values:
        print(f"❌ '{title}' not found in the database.")
        return

    idx = df[df['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    print(f"\n🎬 Because you liked *{title}*, you might also enjoy:")
    for i, score in sim_scores[1:top_n+1]:
        print(f"👉 {df['title'][i]}")

recommend('Inception')