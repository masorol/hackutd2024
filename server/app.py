from flask import Flask, request, jsonify
from flask_cors import CORS
import praw
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Initialize Flask app
app = Flask(__name__)

# Initialize PRAW with your Reddit credentials
reddit = praw.Reddit()

# Initialize SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
analyzer = SentimentIntensityAnalyzer()

# Route to analyze posts by keyword
@app.route('/analyze_topic', methods=['POST'])
def analyze_topic():
    data = request.json
    keyword = data.get('keyword')
    if not keyword:
        return jsonify({"error": "No keyword provided"}), 400
    
    try:
        # Search Reddit posts by keyword (using the 'all' subreddit to search across Reddit)
        submissions = reddit.subreddit('all').search(keyword, limit=100)  # Adjust 'limit' as necessary
        
        # Store sentiment scores and labels
        sentiments = {"positive": 0, "negative": 0, "neutral": 0, "compound": 0}
        total_score = 0
        post_count = 0

        for submission in submissions:
            title = submission.title
            selftext = submission.selftext
            text = title + " " + selftext

            # Perform sentiment analysis on the combined title and selftext
            sentiment = analyzer.polarity_scores(text)
            total_score += sentiment['compound']
            post_count += 1

            # Categorize sentiment
            if sentiment['compound'] >= 0.05:
                sentiments['positive'] += 1
            elif sentiment['compound'] <= -0.05:
                sentiments['negative'] += 1
            else:
                sentiments['neutral'] += 1
        
        # Calculate average sentiment score if there are posts analyzed
        if post_count > 0:
            average_score = total_score / post_count
        else:
            average_score = 0

        return jsonify({
            "total_posts": post_count,
            "sentiments": sentiments,
            "average_sentiment_score": average_score
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    CORS(app)
    app.run(debug=True)