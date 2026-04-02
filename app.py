# ======================================================
# AI Diary : Streamlit Application
#
# This app loads the fine tuned RoBERTa emotion classification model and
# provides a conversational diary interface. Users write freely about how
# they feel, the model detects emotions from the text, and the system
# generates a short empathetic reflection in response.
#
# How to run:
# cd "/Users/faisal/Desktop/Final project/final year project fyp/Final year code"
# streamlit run app.py
 
import streamlit as st
from datetime import datetime
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter
import re
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

# ======================================================
# Page configuration
st.set_page_config(
    page_title="AI Diary",
    page_icon="📔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================================================
# Path to the saved RoBERTa model
MODEL_PATH = "/Users/faisal/Desktop/Final project/final year project fyp/Final year code/roberta_final_model"
 
# ======================================================
# GoEmotions emotion labels, must match the training order
EMOTIONS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral"
]
# ======================================================
# Reflection templates, one per GoEmotions emotion

TEMPLATES = {
    "admiration":    "It is clear that something or someone has genuinely impressed you. That kind of appreciation is worth holding onto.",
    "amusement":     "It is good to hear that something brought a smile today. Moments of lightness are always worth noticing.",
    "anger":         "It makes sense that you are feeling frustrated right now. Your feelings are valid, and acknowledging them is always the right first step.",
    "annoyance":     "Small frustrations can build up if we do not give them space. Recognising them rather than pushing them aside is actually a healthy response.",
    "approval":      "There is something genuinely positive in what you have shared. It sounds like things are moving in a good direction.",
    "caring":        "The fact that you care so deeply says something really positive about you. That kind of empathy is a real strength.",
    "confusion":     "It sounds like things are feeling a bit unclear right now. It is completely okay not to have all the answers immediately.",
    "curiosity":     "That sense of wanting to understand more is a good sign. Curiosity is what keeps us growing and moving forward.",
    "desire":        "Knowing what you want is an important step. Sitting with that feeling can help clarify what to do next.",
    "disappointment":"It sounds like something has not gone the way you hoped. Disappointment is a sign that you cared about the outcome, which is not a bad thing.",
    "disapproval":   "It sounds like something has not sat right with you. Trusting that instinct is worth paying attention to.",
    "disgust":       "Whatever prompted this feeling, your reaction is understandable. It is okay to feel strongly about things that matter to you.",
    "embarrassment": "Those moments can feel really uncomfortable in the moment. But they pass, and they rarely define us the way we fear they might.",
    "excitement":    "That energy really comes through in what you have written. It is great to have something to look forward to.",
    "fear":          "Feeling anxious or uncertain is something a lot of people experience without talking about it. You do not have to have all the answers right now.",
    "gratitude":     "There is something really grounding about noticing what you are grateful for. That perspective takes effort and it is always worth it.",
    "grief":         "Grief is one of the heaviest feelings there is. Whatever you have lost, your feelings about it are completely valid.",
    "joy":           "There is something really uplifting in what you have shared. Hold onto that, these moments matter more than we often give them credit for.",
    "love":          "What you have written carries a lot of warmth. That kind of connection is one of the most meaningful things there is.",
    "nervousness":   "Feeling nervous is often a sign that something matters to you. That is not a weakness, it is a signal worth listening to.",
    "optimism":      "There is a genuine sense of hope in what you have written. That outlook can be a real anchor when things feel uncertain.",
    "pride":         "You should sit with that feeling for a moment. Recognising your own effort and progress is something worth doing.",
    "realization":   "It sounds like something has clicked for you. Those moments of clarity can be really significant, even when they arrive quietly.",
    "relief":        "It is clear that a weight has been lifted. Allow yourself to enjoy that feeling, you have earned it.",
    "remorse":       "Feeling remorse means you have a strong sense of what matters to you. That self-awareness is a foundation you can build on.",
    "sadness":       "It sounds like things have been quite heavy recently. That is completely okay,cd  sometimes sitting with difficult feelings is part of processing them.",
    "surprise":      "It sounds like something caught you off guard. Give yourself a moment to settle before deciding how to respond.",
    "neutral":       "Thank you for taking the time to write this down. Sometimes putting thoughts into words is valuable in itself.",
}
 
# ======================================================
# Wellbeing tips
TIPS = {
    "sadness":    "Try a short walk outside, even 10 minutes of fresh air can gently shift your mood.",
    "anger":      "Box breathing can help, breathe in for 4 counts, hold for 4, out for 4, hold for 4.",
    "fear":       "The 5-4-3-2-1 grounding technique can help, name 5 things you can see right now.",
    "nervousness":"Progressive muscle relaxation, tense each muscle group for 5 seconds then release.",
    "joy":        "Consider writing down what made today good, anchoring positive moments is worth doing.",
    "excitement": "Channel that energy into something creative or productive right now.",
    "gratitude":  "Even three things a day in a gratitude log makes a real difference over time.",
    "grief":      "Be gentle with yourself today. Rest and warmth are valid choices.",
    "neutral":    "Sometimes a quiet moment with no agenda is exactly what the mind needs.",
    "relief":     "Take a deep breath and let yourself fully relax, you have earned this.",
    "pride":      "Consider sharing this feeling with someone who matters to you.",
}
 
# ====================================================== 
# CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
#MainMenu,footer,header{visibility:hidden;}
.block-container{padding-top:0.5rem!important;padding-bottom:1rem;}
section[data-testid="stSidebar"]{background:#0a0f1e;border-right:1px solid rgba(99,102,241,0.2);}
.disclaimer{background:linear-gradient(135deg,#1e3a5f,#1a2d4a);border:1px solid #2d5a8e;border-radius:8px;padding:9px 16px;color:#90caf9;font-size:12px;font-weight:500;text-align:center;margin-bottom:14px;}
.app-title{font-size:22px;font-weight:800;background:linear-gradient(135deg,#60a5fa,#a78bfa);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:2px;}
.app-sub{font-size:11px;color:#475569;margin-bottom:14px;}
.user-pill{background:linear-gradient(135deg,rgba(99,102,241,0.2),rgba(139,92,246,0.2));border:1px solid rgba(99,102,241,0.3);border-radius:20px;padding:5px 14px;font-size:12px;color:#93c5fd;font-weight:600;display:inline-block;margin-bottom:14px;}
.page-header{margin-bottom:18px;padding:16px 0 8px 0;}
.header-title{font-size:26px;font-weight:800;background:linear-gradient(135deg,#60a5fa,#a78bfa,#60a5fa);-webkit-background-clip:text;-webkit-text-fill-color:transparent;}
.header-sub{font-size:13px;color:#64748b;margin-top:2px;}
.msg-wrap-user{display:flex;justify-content:flex-end;margin:8px 0;}
.msg-wrap-ai{display:flex;justify-content:flex-start;margin:8px 0;}
.msg-user{background:linear-gradient(135deg,#6366f1,#8b5cf6);color:white;border-radius:18px 18px 4px 18px;padding:12px 16px;max-width:78%;font-size:14px;line-height:1.6;font-weight:500;box-shadow:0 4px 20px rgba(99,102,241,0.3);}
.msg-ai{background:linear-gradient(135deg,rgba(30,41,59,0.95),rgba(15,23,42,0.95));border:2px solid rgba(99,102,241,0.25);border-radius:18px 18px 18px 4px;padding:14px 18px;max-width:78%;font-size:14px;line-height:1.6;color:#e2e8f0;box-shadow:0 4px 20px rgba(0,0,0,0.3);}
.ai-header{display:flex;align-items:center;gap:6px;margin-bottom:8px;padding-bottom:8px;border-bottom:1px solid rgba(99,102,241,0.2);}
.ai-dot{width:18px;height:18px;background:linear-gradient(135deg,#6366f1,#8b5cf6);border-radius:50%;display:inline-flex;align-items:center;justify-content:center;font-size:9px;color:white;}
.ai-lbl{font-size:10px;font-weight:700;background:linear-gradient(135deg,#60a5fa,#a78bfa);-webkit-background-clip:text;-webkit-text-fill-color:transparent;letter-spacing:0.5px;}
.emotion-tag{display:inline-block;background:linear-gradient(135deg,#3b82f6,#8b5cf6);color:white;border-radius:20px;padding:3px 10px;font-size:11px;font-weight:600;margin:2px;text-transform:capitalize;}
.reflection-box{background:linear-gradient(135deg,rgba(59,130,246,0.12),rgba(139,92,246,0.12));border:1px solid rgba(99,102,241,0.35);border-radius:10px;padding:13px 15px;color:#c7d2fe;font-size:13px;line-height:1.7;font-style:italic;margin-top:10px;}
.tip-box{margin-top:8px;font-size:12px;color:#64748b;}
.login-title{font-size:38px;font-weight:800;text-align:center;background:linear-gradient(135deg,#60a5fa,#a78bfa);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:6px;}
.login-sub{font-size:14px;color:#64748b;text-align:center;margin-bottom:24px;}
</style>
""", unsafe_allow_html=True)
 
# ====================================================== 
# Load the trained RoBERTa mode
# cache_resource loads the model once and keeps it in memory for the session
@st.cache_resource(show_spinner="Loading AI model : about 30 seconds on first run...")
def load_model():
    try:
        tokeniser = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        return tokeniser, model, device, True
    except Exception:
        # If model folder not found, fall back to keyword detection
        return None, None, None, False
 
# ======================================================
# Keyword fallback — used only if model cannot be loaded
EMOTION_KEYWORDS = {
    "joy":           r"\b(happy|happiness|great|amazing|wonderful|excited|good|fantastic|brilliant|glad|pleased|delighted|thrilled|joyful|cheerful|fun|enjoy)\b",
    "sadness":       r"\b(sad|unhappy|depressed|down|miserable|cry|crying|tears|upset|heartbroken|broken|grief|miss|missing|lost|alone|lonely|hurt|pain)\b",
    "anger":         r"\b(angry|anger|furious|mad|rage|hate|frustrated|frustration|annoyed|irritated|livid|outraged|fed up)\b",
    "fear":          r"\b(scared|afraid|terrified|anxious|anxiety|nervous|worry|worried|panic|frightened|dread|dreading)\b",
    "nervousness":   r"\b(nervous|anxious|anxiety|stress|stressed|tense|uneasy|apprehensive|jittery|overwhelmed|pressure)\b",
    "disappointment":r"\b(disappointed|disappointment|let down|failed|failure|expected more|hoped)\b",
    "gratitude":     r"\b(grateful|thankful|thank you|thanks|appreciate|appreciated|blessing|blessed|fortunate|lucky)\b",
    "excitement":    r"\b(excited|exciting|can't wait|looking forward|thrilled|pumped|enthusiastic|eager)\b",
    "love":          r"\b(love|loved|loving|adore|cherish|devoted|affection|romantic|relationship|partner)\b",
    "optimism":      r"\b(hopeful|hope|optimistic|positive|things will|get better|believe|faith|confident)\b",
    "relief":        r"\b(relieved|relief|finally|weight lifted|stress gone|sorted|resolved|done now|finished)\b",
    "pride":         r"\b(proud|pride|accomplished|achieved|achievement|did it|success|succeeded|nailed it)\b",
    "curiosity":     r"\b(curious|wonder|wondering|interesting|fascinated|want to know|learn|explore|discover)\b",
    "confusion":     r"\b(confused|confusing|don't understand|unclear|not sure|unsure|uncertain|puzzled|baffled)\b",
    "surprise":      r"\b(surprised|shocking|shocked|unexpected|didn't expect|caught off guard|unbelievable|wow)\b",
    "remorse":       r"\b(sorry|regret|regretful|guilty|guilt|apologise|apologize|shouldn't have|my fault)\b",
    "neutral":       r"\b(okay|fine|alright|not much|nothing|boring|average|normal|usual|routine|regular)\b",
}
 
def keyword_fallback(text):
    text_lower = text.lower()
    scores = {}
    for emotion, pattern in EMOTION_KEYWORDS.items():
        matches = re.findall(pattern, text_lower)
        if matches:
            scores[emotion] = len(matches)
    if not scores:
        scores["neutral"] = 1
    sorted_emos = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    emotions = [e for e, _ in sorted_emos]
    return emotions[:5], emotions[0]
 
# ======================================================
# Emotion detection
def detect_emotions(text, threshold=0.3):
    """
    Runs text through the RoBERTa model.
    Sigmoid converts logits to per-emotion probabilities.
    Any emotion above the threshold (0.3) is predicted as present.
    The primary emotion is the one with the highest probability.
    """
    tokeniser, model, device, loaded = load_model()
    if not loaded:
        return keyword_fallback(text)
 
    inputs = tokeniser(
        str(text), return_tensors="pt",
        truncation=True, padding=True, max_length=128
    ).to(device)
 
    with torch.no_grad():
        logits = model(**inputs).logits
 
    probs = torch.sigmoid(logits).squeeze().cpu().numpy()
    hits = [(EMOTIONS[i], float(probs[i])) for i, p in enumerate(probs) if p >= threshold]
 
    if not hits:
        hits = [(EMOTIONS[int(np.argmax(probs))], float(np.max(probs)))]
 
    hits.sort(key=lambda x: x[1], reverse=True)
    names = [e for e, _ in hits]
    return names[:5], names[0]
 
# ====================================================== 
# Build AI response HTML
def build_response(text):
    emos, primary = detect_emotions(text)
    reflection = TEMPLATES.get(primary, TEMPLATES["neutral"])
    tip = TIPS.get(primary, "")
    tags = "".join(f'<span class="emotion-tag">{e}</span>' for e in emos)
    html = (
        f'<div class="ai-header"><span class="ai-dot">✦</span>'
        f'<span class="ai-lbl">AI COMPANION</span></div>'
        f'<div style="margin-bottom:9px;">'
        f'<span style="font-size:11px;color:#475569;font-weight:500;">Detected emotions</span><br>'
        f'{tags}</div>'
        f'<div class="reflection-box">💬 {reflection}</div>'
    )
    if tip:
        html += f'<div class="tip-box">{tip}</div>'
    return html, emos, primary
 
 
def show_disclaimer():
    st.markdown(
        '<div class="disclaimer">🔒 This is a prototype application. '
        'Your data will not be accessed, used, or shared. '
        'This is purely for demonstration purposes.</div>',
        unsafe_allow_html=True
    )
 
# ====================================================== 
# Session state
for key, default in {
    "logged_in": False, "username": "", "page": "Chat",
    "entries": [], "users": {}, "chat_history": []
}.items():
    if key not in st.session_state:
        st.session_state[key] = default
 
 
# ====================================================== 
# LOGIN PAGE

def login_page():
    show_disclaimer()
    st.markdown('<div class="login-title">📔 AI Diary</div>', unsafe_allow_html=True)
    st.markdown('<div class="login-sub">Your private space to reflect, understand, and grow.</div>', unsafe_allow_html=True)
 
    _, col, _ = st.columns([1, 1.5, 1])
    with col:
        tab_in, tab_up = st.tabs(["Sign In", "Create Account"])
 
        with tab_in:
            with st.form("login_form"):
                u = st.text_input("Username", placeholder="Enter your username")
                p = st.text_input("PIN", type="password", max_chars=4, placeholder="4-digit PIN")
                if st.form_submit_button("Sign In →", use_container_width=True, type="primary"):
                    if u and p:
                        if u in st.session_state.users and st.session_state.users[u] == p:
                            st.session_state.logged_in = True
                            st.session_state.username = u
                            st.rerun()
                        else:
                            st.error("Incorrect username or PIN.")
                    else:
                        st.warning("Please fill in both fields.")
 
        with tab_up:
            with st.form("signup_form"):
                nu = st.text_input("Username", placeholder="Choose a username")
                np_ = st.text_input("PIN", type="password", max_chars=4, placeholder="Choose a 4-digit PIN")
                cp = st.text_input("Confirm PIN", type="password", max_chars=4, placeholder="Confirm PIN")
                if st.form_submit_button("Create Account →", use_container_width=True, type="primary"):
                    if not nu or not np_:
                        st.warning("Fill in all fields.")
                    elif not np_.isdigit() or len(np_) != 4:
                        st.error("PIN must be exactly 4 digits.")
                    elif np_ != cp:
                        st.error("PINs do not match.")
                    elif nu in st.session_state.users:
                        st.error("That username is already taken.")
                    else:
                        st.session_state.users[nu] = np_
                        st.session_state.logged_in = True
                        st.session_state.username = nu
                        st.rerun()
 
 
# ======================================================
# SIDEBAR

def show_sidebar():
    with st.sidebar:
        st.markdown('<div class="app-title">📔 AI Diary</div>', unsafe_allow_html=True)
        st.markdown('<div class="app-sub">Your reflection companion</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="user-pill">👤 {st.session_state.username}</div>', unsafe_allow_html=True)
        st.markdown("---")
 
        for page_key, label in {
            "Chat": "💬  Daily Reflection",
            "Entries": "📖  Journal Entries",
            "Analytics": "📊  Analytics"
        }.items():
            btn_type = "primary" if st.session_state.page == page_key else "secondary"
            if st.button(label, key=f"nav_{page_key}", use_container_width=True, type=btn_type):
                st.session_state.page = page_key
                st.rerun()
 
        st.markdown("---")
        if st.button("🚪  Log Out", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.username = ""
            st.session_state.chat_history = []
            st.rerun()
 
        st.markdown(
            '<div style="font-size:10px;color:#1e293b;text-align:center;margin-top:20px;line-height:1.6;">'
            '🔒 Prototype only.<br>No data stored or shared.</div>',
            unsafe_allow_html=True
        )
 
 
# ======================================================
# CHAT PAGE

def chat_page():
    show_disclaimer()
    st.markdown(
        '<div class="page-header">'
        '<div class="header-title">✦ Daily Reflection</div>'
        '<div class="header-sub">✨ Share your thoughts, feelings, and experiences</div>'
        '</div>',
        unsafe_allow_html=True
    )
 
    if not st.session_state.chat_history:
        st.session_state.chat_history = [{
            "role": "assistant",
            "html": (
                '<div class="ai-header"><span class="ai-dot">✦</span>'
                '<span class="ai-lbl">AI COMPANION</span></div>'
                'Welcome to your AI Diary. I am here to help you reflect on your '
                'thoughts, feelings, and experiences. How are you feeling today?'
            ),
            "text": "Welcome"
        }]
 
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="msg-wrap-user"><div class="msg-user">{msg["text"]}</div></div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="msg-wrap-ai"><div class="msg-ai">{msg["html"]}</div></div>',
                unsafe_allow_html=True
            )
 
    st.markdown("<br>", unsafe_allow_html=True)
 
    col_text, col_btn = st.columns([11, 1])
    with col_text:
        user_input = st.text_area(
            "message", label_visibility="collapsed",
            placeholder="💭 Share your thoughts, feelings, or experiences...",
            height=80, key="chat_input"
        )
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        send = st.button("➤", use_container_width=True, type="primary")
 
    st.markdown(
        '<div style="font-size:11px;color:#334155;margin-top:2px;">Press the button to send</div>',
        unsafe_allow_html=True
    )
 
    if send and user_input and user_input.strip():
        st.session_state.chat_history.append({"role": "user", "html": user_input, "text": user_input})
 
        with st.spinner("✦ Detecting emotions..."):
            response_html, emos, primary = build_response(user_input)
 
        st.session_state.chat_history.append({"role": "assistant", "html": response_html, "text": response_html})
        st.session_state.entries.insert(0, {
            "username":  st.session_state.username,
            "text":      user_input,
            "emotions":  emos,
            "primary":   primary,
            "timestamp": datetime.now().strftime("%d %b %Y, %H:%M"),
            "date":      datetime.now().strftime("%Y-%m-%d"),
        })
        st.rerun()
 
 
# ======================================================
# ENTRIES PAGE

def entries_page():
    show_disclaimer()
    st.markdown(
        '<div class="page-header">'
        '<div class="header-title">📖 Journal Entries</div>'
        '<div class="header-sub">Browse and review your past reflections</div>'
        '</div>',
        unsafe_allow_html=True
    )
 
    ue = [e for e in st.session_state.entries if e["username"] == st.session_state.username]
    if not ue:
        st.info("No entries yet. Start a conversation in Daily Reflection to create your first entry.")
        return
 
    primaries = sorted(set(e["primary"] for e in ue))
    filt = st.selectbox("Filter by emotion", ["All"] + primaries, label_visibility="collapsed")
    filtered = ue if filt == "All" else [e for e in ue if e["primary"] == filt]
 
    st.markdown(
        f'<div style="font-size:12px;color:#475569;margin-bottom:12px;">{len(filtered)} entries</div>',
        unsafe_allow_html=True
    )
 
    for entry in filtered:
        tags = "".join(f'<span class="emotion-tag">{em}</span>' for em in entry["emotions"][:4])
        with st.expander(f"📅  {entry['timestamp']}  ·  {entry['primary'].capitalize()}"):
            st.markdown(
                f'<div style="font-size:13px;color:#cbd5e1;line-height:1.6;">{entry["text"]}</div>',
                unsafe_allow_html=True
            )
            st.markdown(f'<div style="margin-top:8px;">{tags}</div>', unsafe_allow_html=True)
            ref = TEMPLATES.get(entry["primary"], TEMPLATES["neutral"])
            st.markdown(f'<div class="reflection-box">💬 {ref}</div>', unsafe_allow_html=True)
 
 
# ======================================================
# ANALYTICS PAGE

def analytics_page():
    show_disclaimer()
    st.markdown(
        '<div class="page-header">'
        '<div class="header-title">📊 Analytics</div>'
        '<div class="header-sub">Your emotional patterns at a glance</div>'
        '</div>',
        unsafe_allow_html=True
    )
 
    ue = [e for e in st.session_state.entries if e["username"] == st.session_state.username]
    if len(ue) < 2:
        st.info("Add at least 2 entries through the Daily Reflection page to see your analytics.")
        return
 
    df = pd.DataFrame(ue)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Entries", len(ue))
    c2.metric("Unique Emotions", len(set(e["primary"] for e in ue)))
    c3.metric("Top Emotion", df["primary"].value_counts().index[0].capitalize())
    c4.metric("Session Total", len(ue))
 
    st.markdown("---")
    cl, cr = st.columns(2)
 
    with cl:
        st.markdown("**Emotion Frequency**")
        all_emos = []
        for e in ue:
            all_emos.extend(e["emotions"])
        top8 = dict(sorted(Counter(all_emos).items(), key=lambda x: x[1], reverse=True)[:8])
        fig, ax = plt.subplots(figsize=(5, 4))
        fig.patch.set_facecolor('#0a0f1e')
        ax.set_facecolor('#0a0f1e')
        ax.barh(list(top8.keys())[::-1], list(top8.values())[::-1], color='#6366f1', edgecolor='none', height=0.6)
        ax.tick_params(colors='#94a3b8', labelsize=9)
        for s in ax.spines.values():
            s.set_color('#1e293b')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='x', color='#1e293b', linewidth=0.8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
 
    with cr:
        st.markdown("**Primary Emotion Distribution**")
        pc = df["primary"].value_counts()
        colors = ['#60a5fa', '#a78bfa', '#34d399', '#f87171', '#fbbf24', '#818cf8', '#f472b6', '#2dd4bf']
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        fig2.patch.set_facecolor('#0a0f1e')
        ax2.set_facecolor('#0a0f1e')
        ax2.pie(pc.values, labels=pc.index, autopct='%1.0f%%',
                colors=colors[:len(pc)], textprops={'color': '#94a3b8', 'fontsize': 9}, pctdistance=0.82)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()
 
    if len(ue) >= 3:
        st.markdown("---")
        st.markdown("**Emotion Timeline**")
        tl = pd.DataFrame([
            {"Entry": i + 1, "Emotion": e["primary"].capitalize(), "Time": e["timestamp"]}
            for i, e in enumerate(reversed(ue))
        ])
        st.dataframe(tl, use_container_width=True, hide_index=True)
 
 
# ======================================================
# MAIN

def main():
    if not st.session_state.logged_in:
        login_page()
        return
    show_sidebar()
    page = st.session_state.page
    if page == "Chat":
        chat_page()
    elif page == "Entries":
        entries_page()
    elif page == "Analytics":
        analytics_page()
 
 
if __name__ == "__main__":
    main()