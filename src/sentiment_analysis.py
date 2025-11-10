from textblob import TextBlob

def add_sentiment(df):
    df['sentiment_polarity'] = df['description'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    df['sentiment'] = df['sentiment_polarity'].apply(
        lambda x: 'Positive' if x > 0.1 else ('Negative' if x < -0.1 else 'Neutral')
    )
    return df
