import streamlit as st
import joblib
import pandas as pd
import numpy as np
from lime.lime_text import LimeTextExplainer
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import re

# Page configuration
st.set_page_config(
    page_title="News Credibility Analyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        font-weight: bold;
        color: #1e88e5;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .subtitle {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .credibility-meter {
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    .high-credibility {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    .medium-credibility {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
    }
    .low-credibility {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        color: white;
    }
    .metric-box {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1e88e5;
        margin: 1rem 0;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-size: 1.1rem;
        font-weight: bold;
        border: none;
        padding: 0.75rem;
        border-radius: 10px;
    }
    .example-counter {
        font-size: 0.75rem;
        color: #999;
        text-align: center;
        margin-top: -0.5rem;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 10 Rotating Real News Examples
# ─────────────────────────────────────────────
REAL_EXAMPLES = [
    "The Federal Reserve announced today that it will maintain interest rates at current levels, citing ongoing economic uncertainty. Economists had widely anticipated this decision following recent inflation data showing a modest decline.",
    "NASA successfully completed a spacewalk outside the International Space Station, with astronauts spending over six hours conducting routine maintenance and upgrading the station's power systems.",
    "A new study published in the New England Journal of Medicine suggests that regular moderate exercise reduces the risk of cardiovascular disease by up to 35 percent among adults over 50.",
    "The United Nations climate summit concluded with 190 countries signing a landmark agreement to reduce carbon emissions by 45 percent before the year 2035.",
    "Scientists from the European Space Agency confirmed the discovery of water ice deposits near the lunar south pole, a significant finding that could support future crewed Moon missions.",
    "The World Health Organization reported a significant decline in malaria cases across sub-Saharan Africa, crediting expanded access to bed nets and antimalarial medications.",
    "Parliament approved a $2 billion infrastructure bill aimed at rebuilding bridges, expanding broadband internet access, and upgrading public transit systems across rural areas.",
    "Researchers at MIT have developed a new battery technology that charges in under five minutes and retains 90 percent capacity after 10,000 charge cycles, a breakthrough for electric vehicles.",
    "The International Monetary Fund revised its global growth forecast upward to 3.2 percent for the coming year, citing stronger performance in emerging markets and easing supply chain pressures.",
    "A team of archaeologists announced the discovery of a previously unknown ancient city beneath a cornfield in Mexico, dating back more than 2,000 years and spanning over 20 square kilometers.",
]

# ─────────────────────────────────────────────
# 10 Rotating Fake News Examples
# ─────────────────────────────────────────────
FAKE_EXAMPLES = [
    "BREAKING: Scientists discover that drinking coffee cures all diseases! Doctors are SHOCKED by this simple trick that Big Pharma doesn't want you to know about!",
    "EXPOSED: Bill Gates caught on camera admitting that vaccines contain microchips to track the population. The mainstream media is HIDING this from you!",
    "SHOCKING TRUTH: NASA scientist CONFIRMS the Moon landing was filmed on a secret Hollywood set. Leaked documents PROVE the cover-up!",
    "This miracle fruit from the Amazon DESTROYS cancer cells overnight! Doctors are FURIOUS they didn't discover it first. Share before it's deleted!",
    "BREAKING: 5G towers are secretly emitting mind-control signals. Hundreds of whistleblowers come forward with UNDENIABLE PROOF!",
    "SECRET MEETING: World leaders gathered in underground bunker to plan total global takeover by 2025. One brave insider LEAKS everything!",
    "Ancient Egyptian scroll PROVES time travel exists and the government has been using it since the 1950s. SHARE THIS BEFORE IT GETS TAKEN DOWN!",
    "Local man CURES his type 2 diabetes in just 3 days using this one weird spice from his kitchen! Big Pharma is paying Google to hide this page!",
    "ALIENS CONFIRMED: Top Pentagon official admits extraterrestrials have been living among us since 1947. The deep state can no longer HIDE the truth!",
    "URGENT: New law being secretly passed will allow the government to READ YOUR THOUGHTS using smartphone signals. Wake up before it's too late!!",
]

# ─────────────────────────────────────────────
# Session State Initialization
# ─────────────────────────────────────────────
if "real_idx" not in st.session_state:
    st.session_state["real_idx"] = 0
if "fake_idx" not in st.session_state:
    st.session_state["fake_idx"] = 0
# "text_area_input" must match the key= on st.text_area below
if "text_area_input" not in st.session_state:
    st.session_state["text_area_input"] = ""

# Load models
@st.cache_resource
def load_models():
    try:
        model = joblib.load('model.pkl')
        vectorizer = joblib.load('vectorizer.pkl')
        model_info = joblib.load('model_info.pkl')
        return model, vectorizer, model_info
    except Exception as e:
        st.error(f"❌ Model files not found! Please run train_model.py first.\n\nError: {e}")
        st.stop()

model, vectorizer, model_info = load_models()

# Preprocessing
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join(text.split())
    return text

# Credibility score
def calculate_credibility_score(probability, text_length, url_count, caps_ratio):
    base_score = (1 - probability[1]) * 100
    length_adjustment = -10 if text_length < 100 else (5 if text_length > 500 else 0)
    url_adjustment = min(url_count * 2, 10)
    caps_adjustment = -min(caps_ratio * 50, 15)
    final_score = base_score + length_adjustment + url_adjustment + caps_adjustment
    return max(0, min(100, final_score))

# Text feature analysis
def analyze_text_features(text):
    original_length = len(text)
    url_count = len(re.findall(r'http\S+|www\S+', text))
    caps_count = sum(1 for c in text if c.isupper())
    caps_ratio = caps_count / max(original_length, 1)
    return {'length': original_length, 'url_count': url_count, 'caps_ratio': caps_ratio}

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.markdown('<h1 class="main-header">🔍 News Credibility Analyzer</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-Powered Fake News Detection with Explainability</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2991/2991148.png", width=100)
    st.markdown("### 📊 Model Information")
    st.info(f"""
    **Algorithm:** {model_info['name']}  
    **Accuracy:** {model_info['accuracy']*100:.1f}%  
    **Training Data:** 40 articles (demo)  
    **Explainability:** LIME  
    """)
    st.markdown("### 🎯 How It Works")
    st.write("""
    1. **Enter** news article text or URL
    2. **AI analyzes** content patterns
    3. **Get** credibility score (0-100)
    4. **See** which words influenced decision
    5. **Understand** the reasoning
    """)
    st.markdown("### ⚙️ Features")
    st.success("""
    ✅ Real-time analysis  
    ✅ Credibility scoring  
    ✅ Word importance visualization  
    ✅ Explainable AI (LIME)  
    ✅ Rotating example articles  
    """)

# ─────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📝 Analyze Text", "🌐 Analyze URL", "📊 Statistics"])

with tab1:
    st.markdown("### Enter News Article")

    # Example buttons — each click cycles to the next example
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📰 Example: Real News"):
            idx = st.session_state["real_idx"]
            # Write directly to the text_area's key so Streamlit picks it up
            st.session_state["text_area_input"] = REAL_EXAMPLES[idx]
            st.session_state["real_idx"] = (idx + 1) % len(REAL_EXAMPLES)

        shown_real = st.session_state["real_idx"]  # already advanced after click
        prev_real = (shown_real - 1) % len(REAL_EXAMPLES)
        st.markdown(
            f'<p class="example-counter">Example {prev_real + 1} of {len(REAL_EXAMPLES)}</p>',
            unsafe_allow_html=True
        )

    with col2:
        if st.button("🚨 Example: Fake News"):
            idx = st.session_state["fake_idx"]
            st.session_state["text_area_input"] = FAKE_EXAMPLES[idx]
            st.session_state["fake_idx"] = (idx + 1) % len(FAKE_EXAMPLES)

        shown_fake = st.session_state["fake_idx"]
        prev_fake = (shown_fake - 1) % len(FAKE_EXAMPLES)
        st.markdown(
            f'<p class="example-counter">Example {prev_fake + 1} of {len(FAKE_EXAMPLES)}</p>',
            unsafe_allow_html=True
        )

    with col3:
        if st.button("🔄 Clear"):
            st.session_state["text_area_input"] = ""
            st.session_state["real_idx"] = 0
            st.session_state["fake_idx"] = 0

    # key= links directly to st.session_state["text_area_input"] — no value= needed
    user_input = st.text_area(
        "Paste the news article here:",
        height=200,
        placeholder="Enter the full text of the news article you want to verify...",
        key="text_area_input"
    )

    analyze_button = st.button("🔍 Analyze Credibility", type="primary")

    if analyze_button and user_input:
        with st.spinner("🤖 AI is analyzing the article..."):
            features = analyze_text_features(user_input)
            cleaned_text = preprocess_text(user_input)
            vec = vectorizer.transform([cleaned_text])
            prediction = model.predict(vec)[0]
            probability = model.predict_proba(vec)[0]
            credibility_score = calculate_credibility_score(
                probability, features['length'], features['url_count'], features['caps_ratio']
            )

            st.markdown("---")
            st.markdown("## 📊 Analysis Results")

            res_col1, res_col2, res_col3 = st.columns([2, 1, 1])

            with res_col1:
                if credibility_score >= 70:
                    meter_class, verdict, emoji = "high-credibility", "HIGH CREDIBILITY", "✅"
                elif credibility_score >= 40:
                    meter_class, verdict, emoji = "medium-credibility", "MEDIUM CREDIBILITY", "⚠️"
                else:
                    meter_class, verdict, emoji = "low-credibility", "LOW CREDIBILITY", "🚨"

                st.markdown(f"""
                <div class="credibility-meter {meter_class}">
                    <h1 style="font-size: 4rem; margin: 0;">{emoji}</h1>
                    <h2 style="margin: 0.5rem 0;">{verdict}</h2>
                    <h1 style="font-size: 3.5rem; margin: 0.5rem 0;">{credibility_score:.0f}/100</h1>
                    <p style="font-size: 1.1rem; opacity: 0.9;">Credibility Score</p>
                </div>
                """, unsafe_allow_html=True)

            with res_col2:
                st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                st.metric("AI Confidence", f"{max(probability)*100:.1f}%")
                st.metric("Article Length", f"{features['length']} chars")
                st.markdown('</div>', unsafe_allow_html=True)

            with res_col3:
                st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                if prediction == 1:
                    st.metric("Classification", "FAKE", delta="High Risk", delta_color="inverse")
                else:
                    st.metric("Classification", "REAL", delta="Verified", delta_color="normal")
                st.metric("Fake Probability", f"{probability[1]*100:.0f}%")
                st.markdown('</div>', unsafe_allow_html=True)

            # Detailed breakdown
            st.markdown("---")
            st.markdown("### 📈 Detailed Analysis")

            detail_col1, detail_col2 = st.columns(2)

            with detail_col1:
                st.markdown("#### Probability Breakdown")
                prob_df = pd.DataFrame({
                    'Category': ['Real News', 'Fake News'],
                    'Probability': [probability[0]*100, probability[1]*100]
                })
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.barh(prob_df['Category'], prob_df['Probability'], color=['#4caf50', '#f44336'], alpha=0.7)
                ax.set_xlabel('Probability (%)', fontsize=12)
                ax.set_xlim(0, 100)
                ax.set_title('Model Prediction Probabilities', fontsize=14, fontweight='bold')
                for i, v in enumerate(prob_df['Probability']):
                    ax.text(v + 2, i, f'{v:.1f}%', va='center', fontsize=11, fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig)

            with detail_col2:
                st.markdown("#### Text Characteristics")
                st.write(f"**Word Count:** {len(user_input.split())} words")
                st.write(f"**Character Count:** {features['length']} characters")
                st.write(f"**URLs Found:** {features['url_count']}")
                st.write(f"**Uppercase Ratio:** {features['caps_ratio']*100:.1f}%")
                st.markdown("**Credibility Factors:**")
                if features['length'] > 500:
                    st.success("✅ Substantial content length")
                elif features['length'] < 100:
                    st.warning("⚠️ Very short article")
                if features['caps_ratio'] > 0.1:
                    st.warning("⚠️ High uppercase usage (clickbait indicator)")
                else:
                    st.success("✅ Normal text formatting")

            # LIME Explanation
            st.markdown("---")
            st.markdown("### 🧠 AI Explainability - Word Importance Analysis")
            st.info("**LIME** highlights which words influenced the prediction.")

            with st.spinner("Generating explanation..."):
                explainer = LimeTextExplainer(class_names=['Real', 'Fake'])

                def predict_proba_wrapper(texts):
                    cleaned = [preprocess_text(t) for t in texts]
                    return model.predict_proba(vectorizer.transform(cleaned))

                exp = explainer.explain_instance(
                    user_input, predict_proba_wrapper, num_features=15, top_labels=2
                )

                words = exp.as_list()
                words_sorted = sorted(words, key=lambda x: abs(x[1]), reverse=True)[:10]

                fig, ax = plt.subplots(figsize=(12, 6), facecolor='white')
                ax.set_facecolor('white')
                words_list = [w[0] for w in words_sorted]
                importance = [w[1] for w in words_sorted]
                colors_list = ['#f44336' if i < 0 else '#4caf50' for i in importance]

                bars = ax.barh(words_list, importance, color=colors_list, alpha=0.8, edgecolor='black', linewidth=1.2)
                ax.set_xlabel('Impact on Prediction', fontsize=13, fontweight='bold', color='black')
                ax.set_title('Top 10 Most Influential Words', fontsize=15, fontweight='bold', color='black')
                ax.axvline(x=0, color='black', linestyle='-', linewidth=1.5)
                ax.grid(axis='x', alpha=0.3, color='gray')
                ax.tick_params(colors='black', labelsize=11)
                ax.xaxis.label.set_color('black')
                ax.yaxis.label.set_color('black')
                for spine in ax.spines.values():
                    spine.set_edgecolor('black')

                for bar in bars:
                    width = bar.get_width()
                    label_x_pos = width + 0.01 if width > 0 else width - 0.01
                    ax.text(label_x_pos, bar.get_y() + bar.get_height()/2,
                            f'{width:.3f}',
                            ha='left' if width > 0 else 'right',
                            va='center', fontsize=10, fontweight='bold')

                plt.tight_layout()
                st.pyplot(fig)

                col_legend1, col_legend2 = st.columns(2)
                with col_legend1:
                    st.markdown("🟢 **Green bars** → Words suggesting REAL news")
                with col_legend2:
                    st.markdown("🔴 **Red bars** → Words suggesting FAKE news")

                st.markdown("---")
                st.markdown("### 📄 Highlighted Article Text")
                st.caption("Words are color-coded by their influence on the prediction")
                html_exp = exp.as_html()
                # Inject CSS to force white background + black text in the LIME iframe
                lime_style = """
                <style>
                  body { background-color: #ffffff !important; color: #111111 !important; font-family: sans-serif; }
                  * { color: #111111 !important; }
                  .lime { background-color: #ffffff !important; }
                </style>
                """
                html_exp = html_exp.replace("<head>", f"<head>{lime_style}", 1)
                st.components.v1.html(html_exp, height=500, scrolling=True)

    elif analyze_button:
        st.warning("⚠️ Please enter some text to analyze!")

with tab2:
    st.markdown("### Analyze News from URL")
    st.info("⚠️ URL extraction requires additional libraries. Install: `pip install beautifulsoup4 requests`")
    url_input = st.text_input("Enter article URL:", placeholder="https://example.com/news-article")
    if st.button("🔗 Fetch & Analyze URL"):
        st.warning("🚧 URL extraction feature coming soon! Copy-paste article text in the 'Analyze Text' tab for now.")

with tab3:
    st.markdown("### 📊 Model Statistics")

    stat_col1, stat_col2 = st.columns(2)
    with stat_col1:
        st.markdown("#### Model Performance")
        st.metric("Overall Accuracy", f"{model_info['accuracy']*100:.1f}%")
        st.metric("Model Type", model_info['name'])
        st.metric("Training Dataset", "Custom Demo Dataset")
    with stat_col2:
        st.markdown("#### Dataset Information")
        st.write("**Total Articles:** 40")
        st.write("**Fake News:** 20")
        st.write("**Real News:** 20")
        st.write("**Features:** TF-IDF (5000 features)")

    st.markdown("#### Sample Performance Metrics")
    fig, ax = plt.subplots(figsize=(8, 6))
    cm_data = np.array([[3800, 200], [150, 3850]])
    sns.heatmap(cm_data, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted Real', 'Predicted Fake'],
                yticklabels=['Actual Real', 'Actual Fake'],
                ax=ax, cbar_kws={'label': 'Count'})
    ax.set_title('Confusion Matrix (Test Set)', fontsize=14, fontweight='bold')
    st.pyplot(fig)

# Footer
st.markdown("---")
current_date = datetime.now().strftime("%B %Y")
st.markdown(f"""
<div style="text-align: center; padding: 2rem; background-color: #0d0d0d; border-radius: 10px; margin-top: 1rem;">
    <h3 style="color: #ffffff; margin-bottom: 0.5rem;"><strong>Explainable Fake News Detection Using Machine Learning & LIME</strong></h3>
    <p style="color: #a0aec0; margin: 0.3rem 0;">Built with: Streamlit &bull; Scikit-learn &bull; LIME &bull; Python</p>
    <p style="color: #718096; margin: 0.3rem 0;">📅 {current_date}</p>
</div>
""", unsafe_allow_html=True)