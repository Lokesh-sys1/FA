import pandas as pd
import joblib
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

print("Training Fake News Model...")

# ---------------------------
# REAL NEWS EXAMPLES
# ---------------------------
real_news = [
    "Government announces new economic policy to stabilize inflation.",
    "Scientists discover new species in the Amazon rainforest.",
    "The central bank has decided to maintain interest rates this quarter.",
    "NASA successfully launches satellite to monitor climate change.",
    "Researchers publish study showing benefits of balanced diet.",
    "New infrastructure project aims to improve public transportation.",
    "University researchers develop new vaccine technology.",
    "The parliament passed a bill to support renewable energy.",
    "International summit discusses global economic recovery.",
    "Health officials recommend vaccination to prevent diseases.",
    "Federal Reserve holds interest rates steady amid economic uncertainty.",
    "New climate report warns of rising sea levels by 2050.",
    "World Health Organization issues updated health guidelines.",
    "Stock markets close higher after positive earnings reports.",
    "Scientists confirm evidence of water ice on the Moon's surface.",
    "UN peacekeeping mission extended in conflict-affected region.",
    "New trade agreement signed between major economies.",
    "Supreme Court rules on landmark environmental protection case.",
    "Global renewable energy capacity doubles over the past decade.",
    "Education ministry launches national digital literacy program.",
]

# ---------------------------
# FAKE NEWS EXAMPLES
# ---------------------------
fake_news = [
    "BREAKING: Drinking lemon water cures all diseases instantly!",
    "Scientists shocked after discovering aliens living under Antarctica!",
    "Government secretly controlling weather using hidden machines!",
    "Doctors hate this trick that cures diabetes overnight!",
    "Celebrity reveals secret pill that stops aging forever!",
    "Miracle plant discovered that can regrow human limbs!",
    "Secret government document proves the earth is hollow!",
    "New study claims chocolate makes people invisible!",
    "Ancient pyramid found on Mars by secret space mission!",
    "Magic fruit discovered that increases IQ instantly!",
    "SHOCKING: Bill Gates admits microchips are in all vaccines!",
    "5G towers are causing birds to fall from the sky, scientists say!",
    "NASA admits the Moon landing was staged in a Hollywood studio!",
    "Doctors HATE her! Woman cures cancer with kitchen spice!",
    "TOP SECRET: Government hiding alien bodies in Area 51 basement!",
    "This one weird trick adds 30 years to your life — Big Pharma is furious!",
    "Earthquake machine hidden under Denver airport, whistleblower claims!",
    "Man grows back missing arm after drinking special herbal tea!",
    "Scientists discover that the Sun is actually cold — everything is a lie!",
    "EXPOSED: World leaders are secretly reptiles in disguise!",
]

texts = real_news + fake_news
labels = [0] * len(real_news) + [1] * len(fake_news)  # 0 = real, 1 = fake

df = pd.DataFrame({"text": texts, "label": labels})

def clean(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    return text

df["text"] = df["text"].apply(clean)

X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

pred = model.predict(X_test_vec)
acc = accuracy_score(y_test, pred)
print(f"Accuracy: {acc:.4f}")

# ✅ Save all three required files
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
joblib.dump({
    "name": "Logistic Regression",
    "accuracy": acc
}, "model_info.pkl")

print("✅ model.pkl saved")
print("✅ vectorizer.pkl saved")
print("✅ model_info.pkl saved")
print("Done! Run: streamlit run app.py")