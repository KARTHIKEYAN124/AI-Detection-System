"""
Train AI detection model on Kaggle dataset
Dataset: https://www.kaggle.com/datasets/algozee/ai-generated-vs-human-written-text-dataset
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle
import re
import os

def clean_text(text):
    """Clean and preprocess text"""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def train_model():
    print("=" * 60)
    print("AI TEXT DETECTOR - MODEL TRAINING")
    print("=" * 60)
    
    # ============================================
    # OPTION 1: Load real Kaggle dataset
    # ============================================
    # Download the dataset from Kaggle and place in this folder
    # Then uncomment below:
    
    """
    # Load the Kaggle dataset
    print("\n📂 Loading Kaggle dataset...")
    df = pd.read_csv('ai_human_text_dataset.csv')  # Adjust filename
    
    # Inspect the dataset
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nLabel distribution:")
    print(df['label'].value_counts() if 'label' in df.columns else "Check column names")
    
    # Combine text columns if needed
    if 'title' in df.columns and 'text' in df.columns:
        df['full_text'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
        text_col = 'full_text'
    else:
        text_col = 'text'  # Adjust based on dataset
    
    label_col = 'label'  # Adjust based on dataset
    """
    
    # ============================================
    # OPTION 2: Use synthetic training data
    # (Use this if you haven't downloaded Kaggle dataset yet)
    # ============================================
    print("\n📝 Using synthetic training data...")
    print("   (For better accuracy, download Kaggle dataset)")
    
    # AI-generated text examples (label = 1)
    ai_texts = [
        "It is important to note that artificial intelligence has revolutionized numerous industries. Furthermore, the impact of machine learning extends far beyond initial expectations. Moreover, the multifaceted nature of these technologies underscores their pivotal role in modern society.",
        "In today's rapidly evolving digital landscape, organizations must leverage cutting-edge technologies to maintain a competitive edge. The implementation of robust frameworks facilitates streamlined operations and enhances overall efficiency.",
        "Delving into the realm of natural language processing reveals a plethora of opportunities. The comprehensive analysis of linguistic patterns enables researchers to develop more sophisticated models. It is essential to consider the various parameters that influence model performance.",
        "The paradigm shift towards AI-driven solutions has fundamentally transformed how businesses operate. Subsequently, companies have witnessed unprecedented growth in productivity and innovation. The synergistic relationship between human expertise and machine intelligence cannot be overstated.",
        "Furthermore, the holistic approach to problem-solving necessitates a thorough understanding of underlying mechanisms. Consequently, researchers have developed innovative methodologies to address complex challenges. The aforementioned strategies have proven remarkably effective.",
        "In conclusion, the integration of artificial intelligence in various sectors has yielded substantial benefits. It is worth noting that the ongoing research in this field continues to push boundaries. The transformative potential of these technologies is truly remarkable.",
        "The comprehensive analysis of data reveals intricate patterns that would otherwise remain obscure. It is crucial to acknowledge the multifaceted nature of modern challenges. The implementation of robust strategies is paramount to achieving sustainable success.",
        "Moreover, the exploration of novel concepts in machine learning has opened new avenues for research. The utilization of advanced algorithms facilitates the discovery of hidden insights. In the context of contemporary research, these developments are particularly significant.",
        "Navigating the complex landscape of modern technology requires a nuanced understanding of various disciplines. The convergence of different fields has led to groundbreaking innovations. It is imperative that we continue to explore these intersections.",
        "The intricate tapestry of artificial intelligence encompasses numerous subfields, each contributing to our understanding. From computer vision to natural language processing, the applications are vast and varied. The potential for future developments remains virtually limitless.",
        "Subsequently, the field has witnessed exponential growth, with researchers exploring uncharted territories. The comprehensive nature of these investigations requires interdisciplinary collaboration. Furthermore, the implications of these discoveries extend far beyond the immediate applications.",
        "It is essential to recognize that artificial intelligence is not merely a technological advancement but a fundamental shift in how we approach problem-solving. The holistic integration of these systems into various aspects of society necessitates careful consideration of ethical implications and long-term consequences.",
        "The paramount importance of data quality in machine learning cannot be overstated. Researchers must navigate complex challenges related to data collection, preprocessing, and validation. The aforementioned considerations are crucial for developing reliable and robust models.",
        "In the realm of modern computing, the optimization of algorithms has become increasingly important. The multifaceted approach to algorithm design incorporates various techniques, including parallel processing and distributed computing. These methodologies have proven essential for handling large-scale problems.",
        "The paradigm of cloud computing has fundamentally altered how organizations deploy and manage their infrastructure. The scalability and flexibility offered by cloud platforms enable businesses to adapt quickly to changing market conditions. This transformative approach has become the cornerstone of modern IT strategy.",
        "It is worth noting that the field of data science continues to evolve at a rapid pace. The integration of statistical methods with machine learning techniques has yielded unprecedented insights. Researchers must remain cognizant of the various factors that influence model performance and interpretation.",
        "The comprehensive evaluation of machine learning models requires careful consideration of multiple metrics. Accuracy alone is insufficient; practitioners must also examine precision, recall, and F1 scores. The holistic assessment of model performance ensures robust and reliable predictions.",
        "Furthermore, the ethical implications of artificial intelligence demand careful consideration. As these systems become increasingly integrated into society, it is imperative that developers prioritize transparency, fairness, and accountability. The aforementioned principles are essential for responsible AI development.",
        "The multifaceted nature of cybersecurity challenges requires a comprehensive approach to risk management. Organizations must implement robust security protocols and continuously monitor for potential threats. The dynamic landscape of cyber threats necessitates ongoing adaptation and innovation.",
        "In summary, the rapid advancement of technology has created both opportunities and challenges. The strategic implementation of innovative solutions can drive growth and efficiency. However, it is crucial to balance technological progress with ethical considerations and societal impact.",
    ] * 10  # Repeat to increase dataset size
    
    # Human-written text examples (label = 0)
    human_texts = [
        "So I was walking down the street yesterday and I saw this really weird thing. Like, there was a guy trying to fit an entire couch into a tiny car. Honestly, I couldn't believe my eyes. It was honestly one of the funniest things I've seen in a while.",
        "My dog did the craziest thing this morning. She literally ate my homework and then looked at me like I was the one who did something wrong. I mean, come on! You can't just eat someone's homework and then act all innocent about it.",
        "You know what really bugs me? People who don't use turn signals. Like, it's not that hard! Just flip the little lever thing. Honestly, it's not that complicated. I don't get why some people just refuse to do it.",
        "Okay so basically I tried to make pancakes this morning and it was a complete disaster. The smoke alarm went off, my kitchen looked like a warzone, and somehow I still managed to burn water. Yeah, I burned water. Don't ask me how.",
        "I love going to the beach but honestly the sand thing is so annoying. Like, it gets everywhere and you spend like an hour trying to get it off. Plus my dog always tries to eat stuff there. It's a whole thing.",
        "My friend told me the weirdest story the other day. Apparently she ran into her ex at the grocery store and they just stood there in the cereal aisle for like 20 minutes. Awkward doesn't even begin to cover it lol.",
        "Honestly, I think people make parenting way harder than it needs to be. Like, my mom raised four of us and we turned out fine. Just use common sense and you're good. I don't get why everyone needs a parenting book for everything.",
        "You won't believe what happened at work today. My boss called a meeting that could've been an email. Again. I swear, if I have to sit through another hour of him talking about synergy, I'm gonna lose it. The worst part? We could've just slack'd it.",
        "So my neighbor's cat keeps coming into my yard and I honestly don't know what to do. It's super cute but my dog absolutely hates it. There's like a full turf war happening in my backyard every single day. It's kind of hilarious tbh.",
        "I tried cooking something new last night and it was actually pretty good! I made this pasta thing from a recipe I found online. Followed it exactly and somehow didn't burn anything for once. My roommates were actually impressed which is rare lol.",
        "The weather has been so weird lately. One day it's sunny, the next it's like freezing. I honestly don't know what to wear anymore. I went out in shorts yesterday and almost froze to death. Classic me making bad decisions.",
        "My phone died at the worst possible moment today. I was trying to pay for something and it just went black. Had to stand there looking stupid while the cashier waited. Honestly, I need to start charging my phone before I leave the house.",
        "Went to the movies yesterday and someone literally talked through the entire film. Like, the ENTIRE thing. Who does that? I wanted to say something but I'm not really a confrontational person. Just sat there suffering for two hours. Ugh.",
        "My kids drove me absolutely crazy this weekend. They were fighting over a toy and I was like, you know what, figure it out yourselves. I'm too old for this. Parenting is honestly not for the weak. Respect to all the parents out there, fr.",
        "Tried to fix my sink yesterday because I thought I could handle it. Spoiler alert: I could not. Now there's water everywhere and I'm watching YouTube tutorials on how to fix the thing I broke while trying to fix the original problem. Send help.",
        "My grandma makes the best cookies ever. Like, I'm not even exaggerating. Every time I visit she has a fresh batch ready. I always try to steal some for later but she catches me every single time. That woman is a cookie genius honestly.",
        "I really need to clean my apartment but I keep putting it off. It's not that bad but also it's pretty bad. You know when you just look at the mess and think, nah, not today? Yeah that's been me for like two weeks now.",
        "Got stuck in traffic for like an hour today because someone decided to fender bender in the middle of rush hour. I was late to everything and I couldn't even be mad at the guy honestly, could've happened to anyone. Still annoying though.",
        "My coworker brings the most amazing food to work every single day. I'm so jealous. I bring like a sad sandwich and she brings homemade everything. I need to up my lunch game. Or just befriend her. Either works.",
        "Just got back from vacation and honestly I don't want to go back to work. Is it just me or is the Sunday scaries thing real? I was literally fine yesterday and now I'm dreading Monday. Send coffee and motivation please.",
    ] * 10  # Repeat to increase dataset size
    
    # Create DataFrame
    df = pd.DataFrame({
        'text': ai_texts + human_texts,
        'label': [1] * len(ai_texts) + [0] * len(human_texts)
    })
    
    print(f"\n📊 Dataset size: {len(df)} samples")
    print(f"   AI samples: {len(ai_texts)}")
    print(f"   Human samples: {len(human_texts)}")
    
    # Clean text
    print("\n🧹 Cleaning text...")
    df['clean_text'] = df['text'].apply(clean_text)
    
    # Remove empty texts
    df = df[df['clean_text'].str.len() > 20]
    
    # Split data
    print("\n✂️ Splitting data (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        df['clean_text'], 
        df['label'], 
        test_size=0.2, 
        random_state=42,
        stratify=df['label']
    )
    
    # TF-IDF Vectorization
    print("\n🔢 Creating TF-IDF features...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),  # Unigrams and bigrams
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )
    
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    print(f"   Feature dimensions: {X_train_tfidf.shape[1]}")
    
    # Train Logistic Regression
    print("\n🤖 Training Logistic Regression model...")
    model = LogisticRegression(
        max_iter=1000,
        C=1.0,
        solver='lbfgs',
        random_state=42
    )
    
    model.fit(X_train_tfidf, y_train)
    
    # Evaluate
    print("\n📈 Evaluating model...")
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n{'=' * 60}")
    print(f"✅ MODEL ACCURACY: {accuracy * 100:.2f}%")
    print(f"{'=' * 60}")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Human', 'AI']))
    
    # Get top features
    print("\n🔍 Top 20 AI-indicative words/phrases:")
    feature_names = vectorizer.get_feature_names_out()
    coefficients = model.coef_[0]
    top_ai_idx = np.argsort(coefficients)[-20:][::-1]
    for idx in top_ai_idx:
        print(f"   {feature_names[idx]:<30} (weight: {coefficients[idx]:.3f})")
    
    print("\n🔍 Top 20 Human-indicative words/phrases:")
    top_human_idx = np.argsort(coefficients)[:20]
    for idx in top_human_idx:
        print(f"   {feature_names[idx]:<30} (weight: {coefficients[idx]:.3f})")
    
    # Save model
    print("\n💾 Saving model...")
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    
    print("\n✅ Model saved as 'model.pkl' and 'vectorizer.pkl'")
    print("\n🎉 Training complete! You can now run the web app.")
    
    return accuracy

if __name__ == "__main__":
    train_model()
