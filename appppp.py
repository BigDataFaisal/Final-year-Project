import streamlit as st
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from collections import Counter

st.set_page_config(page_title="AI Diary", page_icon="📔", layout="wide", initial_sidebar_state="expanded")

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

MODEL_PATH = "/content/drive/MyDrive/AI_Diary/roberta_final_model"
EMOTIONS = ["admiration","amusement","anger","annoyance","approval","caring","confusion","curiosity","desire","disappointment","disapproval","disgust","embarrassment","excitement","fear","gratitude","grief","joy","love","nervousness","optimism","pride","realization","relief","remorse","sadness","surprise","neutral"]
TEMPLATES = {
    "sadness":"It sounds like things have been quite heavy recently. That is completely okay — sometimes sitting with difficult feelings is part of processing them.",
    "joy":"There is something really uplifting in what you have shared. Hold onto that — these moments matter more than we often give them credit for.",
    "anger":"It makes sense that you are feeling frustrated. Your feelings are valid, and acknowledging them is always the right first step.",
    "fear":"Feeling anxious or uncertain is something a lot of people experience without talking about it. You do not have to have all the answers right now.",
    "surprise":"It sounds like something caught you off guard. Give yourself a moment to settle before deciding how to respond.",
    "disgust":"Whatever prompted this feeling, your reaction is understandable. It is okay to feel strongly about things that matter to you.",
    "admiration":"It is clear that something or someone has genuinely impressed you. That kind of appreciation is worth noticing.",
    "amusement":"It is good to hear that something brought a smile today. Moments of lightness are always worth holding onto.",
    "annoyance":"Small frustrations can build up if we do not give them space. Acknowledging them rather than pushing them aside is a healthy response.",
    "approval":"There is something really positive in what you have shared. It sounds like things are moving in a good direction.",
    "caring":"The fact that you care so deeply says something really positive about you. That kind of empathy is a genuine strength.",
    "confusion":"It sounds like things are feeling a bit unclear right now. It is okay not to have all the answers immediately.",
    "curiosity":"That sense of wanting to understand more is a good sign. Curiosity is what keeps us growing and moving forward.",
    "desire":"Knowing what you want is an important step. Sitting with that feeling can help clarify what to do next.",
    "disappointment":"It sounds like something has not gone the way you hoped. Disappointment is a sign that you cared about the outcome, which is not a bad thing.",
    "disapproval":"It sounds like something has not sat right with you. Trusting that instinct is worth paying attention to.",
    "embarrassment":"Those moments can feel really uncomfortable in the moment. But they pass, and they rarely define us the way we fear they might.",
    "excitement":"That energy really comes through in what you have written. It is great to have something to look forward to.",
    "gratitude":"There is something really grounding about noticing what you are grateful for. That perspective takes effort and it is always worth it.",
    "grief":"Grief is one of the heaviest feelings there is. Whatever you have lost, your feelings about it are completely valid.",
    "love":"What you have written carries a lot of warmth. That kind of connection is one of the most meaningful things there is.",
    "nervousness":"Feeling nervous is often a sign that something matters to you. That is not a weakness — it is a signal worth listening to.",
    "optimism":"There is a genuine sense of hope in what you have written. That outlook can be a real anchor when things feel uncertain.",
    "pride":"You should sit with that feeling for a moment. Recognising your own effort and progress is something worth doing.",
    "realization":"It sounds like something has clicked for you. Those moments of clarity can be significant, even when they arrive quietly.",
    "relief":"It is clear that a weight has been lifted. Allow yourself to enjoy that feeling — you deserve it.",
    "remorse":"Feeling remorse means you have a strong sense of what matters to you. That self-awareness is a foundation you can build on.",
    "neutral":"Thank you for taking the time to write this down. Sometimes putting thoughts into words is valuable in itself.",
}
TIPS = {
    "sadness":"💙 Try a short walk outside — even 10 minutes of fresh air can gently shift your mood.",
    "anger":"💨 Box breathing can help — breathe in for 4 counts, hold 4, out 4, hold 4.",
    "fear":"🧘 The 5-4-3-2-1 grounding technique can help — name 5 things you can see right now.",
    "nervousness":"🧘 Progressive muscle relaxation — tense each muscle group for 5 seconds then release.",
    "joy":"✨ Consider writing down what made today good — anchoring positive moments is worth doing.",
    "excitement":"⚡ Channel that energy into something creative or productive right now.",
    "gratitude":"📝 Even three things a day in a gratitude log makes a real difference over time.",
    "grief":"🫂 Be gentle with yourself today. Rest and warmth are valid choices.",
    "neutral":"☕ Sometimes a quiet moment with no agenda is exactly what the mind needs.",
}

for k,v in {"logged_in":False,"username":"","page":"Chat","entries":[],"users":{},"chat_history":[]}.items():
    if k not in st.session_state:
        st.session_state[k]=v

@st.cache_resource(show_spinner="Loading AI model...")
def load_model():
    try:
        tok=AutoTokenizer.from_pretrained(MODEL_PATH)
        mdl=AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        mdl.eval()
        dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mdl.to(dev)
        return tok,mdl,dev,True
    except:
        return None,None,None,False

def predict(text,thr=0.3):
    tok,mdl,dev,ok=load_model()
    if not ok:
        return ["joy"],"joy"
    inp=tok(str(text),return_tensors="pt",truncation=True,padding=True,max_length=128).to(dev)
    with torch.no_grad():
        logits=mdl(**inp).logits
    probs=torch.sigmoid(logits).squeeze().cpu().numpy()
    hits=[(EMOTIONS[i],float(probs[i])) for i,p in enumerate(probs) if p>=thr]
    if not hits:
        hits=[(EMOTIONS[int(np.argmax(probs))],float(np.max(probs)))]
    hits.sort(key=lambda x:x[1],reverse=True)
    return [e for e,_ in hits],hits[0][0]

def build_response(text):
    emos,primary=predict(text)
    ref=TEMPLATES.get(primary,TEMPLATES["neutral"])
    tip=TIPS.get(primary,"")
    tags="".join(f'<span class="emotion-tag">{e}</span>' for e in emos[:5])
    html=f"""<div class="ai-header"><span class="ai-dot">✦</span><span class="ai-lbl">AI COMPANION</span></div>
<div style="margin-bottom:9px;"><span style="font-size:11px;color:#475569;font-weight:500;">Detected emotions</span><br>{tags}</div>
<div class="reflection-box">💬 {ref}</div>"""
    if tip:
        html+=f'<div class="tip-box">{tip}</div>'
    return html,emos,primary

def disc():
    st.markdown('<div class="disclaimer">🔒 This is a prototype application. Your data will not be accessed, used, or shared. This is purely for demonstration purposes.</div>',unsafe_allow_html=True)

# ── LOGIN ─────────────────────────────────────────────────────────────────────
def login_page():
    disc()
    st.markdown('<div class="login-title">📔 AI Diary</div>',unsafe_allow_html=True)
    st.markdown('<div class="login-sub">Your private space to reflect, understand, and grow.</div>',unsafe_allow_html=True)
    _,c,_=st.columns([1,1.5,1])
    with c:
        t1,t2=st.tabs(["Sign In","Create Account"])
        with t1:
            with st.form("li"):
                u=st.text_input("Username",placeholder="Enter username")
                p=st.text_input("PIN",type="password",max_chars=4,placeholder="4-digit PIN")
                if st.form_submit_button("Sign In →",use_container_width=True,type="primary"):
                    if u and p:
                        if u in st.session_state.users and st.session_state.users[u]==p:
                            st.session_state.logged_in=True
                            st.session_state.username=u
                            st.rerun()
                        else:
                            st.error("Incorrect username or PIN.")
                    else:
                        st.warning("Please fill in both fields.")
        with t2:
            with st.form("su"):
                nu=st.text_input("Username",placeholder="Choose a username")
                np_=st.text_input("PIN",type="password",max_chars=4,placeholder="Choose a 4-digit PIN")
                cp=st.text_input("Confirm PIN",type="password",max_chars=4,placeholder="Confirm PIN")
                if st.form_submit_button("Create Account →",use_container_width=True,type="primary"):
                    if not nu or not np_:
                        st.warning("Fill in all fields.")
                    elif not np_.isdigit() or len(np_)!=4:
                        st.error("PIN must be exactly 4 digits.")
                    elif np_!=cp:
                        st.error("PINs do not match.")
                    elif nu in st.session_state.users:
                        st.error("Username already taken.")
                    else:
                        st.session_state.users[nu]=np_
                        st.session_state.logged_in=True
                        st.session_state.username=nu
                        st.rerun()

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
def sidebar():
    with st.sidebar:
        st.markdown('<div class="app-title">📔 AI Diary</div>',unsafe_allow_html=True)
        st.markdown('<div class="app-sub">Your reflection companion</div>',unsafe_allow_html=True)
        st.markdown(f'<div class="user-pill">👤 {st.session_state.username}</div>',unsafe_allow_html=True)
        st.markdown("---")
        for key,label in {"Chat":"💬  Daily Reflection","Entries":"📖  Journal Entries","Analytics":"📊  Analytics"}.items():
            t="primary" if st.session_state.page==key else "secondary"
            if st.button(label,key=f"nav_{key}",use_container_width=True,type=t):
                st.session_state.page=key
                st.rerun()
        st.markdown("---")
        if st.button("🚪  Log Out",use_container_width=True):
            st.session_state.logged_in=False
            st.session_state.username=""
            st.session_state.chat_history=[]
            st.rerun()
        st.markdown('<div style="font-size:10px;color:#1e293b;text-align:center;margin-top:20px;line-height:1.6;">🔒 Prototype only.<br>No data stored or shared.</div>',unsafe_allow_html=True)

# ── CHAT ──────────────────────────────────────────────────────────────────────
def chat_page():
    disc()
    st.markdown('<div class="page-header"><div class="header-title">✦ Daily Reflection</div><div class="header-sub">✨ Share your thoughts, feelings, and experiences</div></div>',unsafe_allow_html=True)

    if not st.session_state.chat_history:
        st.session_state.chat_history=[{"role":"assistant","html":"""<div class="ai-header"><span class="ai-dot">✦</span><span class="ai-lbl">AI COMPANION</span></div>Welcome to your AI Diary. I am here to help you reflect on your thoughts, feelings, and experiences. How are you feeling today?""","text":"Welcome"}]

    for msg in st.session_state.chat_history:
        if msg["role"]=="user":
            st.markdown(f'<div class="msg-wrap-user"><div class="msg-user">{msg["text"]}</div></div>',unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="msg-wrap-ai"><div class="msg-ai">{msg["html"]}</div></div>',unsafe_allow_html=True)

    st.markdown("<br>",unsafe_allow_html=True)

    c1,c2=st.columns([11,1])
    with c1:
        user_input=st.text_area("msg",label_visibility="collapsed",placeholder="💭 Share your thoughts, feelings, or experiences...",height=80,key="chat_input")
    with c2:
        st.markdown("<br>",unsafe_allow_html=True)
        send=st.button("➤",use_container_width=True,type="primary")
    st.markdown('<div style="font-size:11px;color:#334155;margin-top:2px;">Press the button to send</div>',unsafe_allow_html=True)

    if send and user_input and user_input.strip():
        st.session_state.chat_history.append({"role":"user","html":user_input,"text":user_input})
        with st.spinner("✦ Thinking..."):
            html,emos,primary=build_response(user_input)
        st.session_state.chat_history.append({"role":"assistant","html":html,"text":html})
        st.session_state.entries.insert(0,{"username":st.session_state.username,"text":user_input,"emotions":emos,"primary":primary,"timestamp":datetime.now().strftime("%d %b %Y, %H:%M"),"date":datetime.now().strftime("%Y-%m-%d")})
        st.rerun()

# ── ENTRIES ───────────────────────────────────────────────────────────────────
def entries_page():
    disc()
    st.markdown('<div class="page-header"><div class="header-title">📖 Journal Entries</div><div class="header-sub">Browse and review your past reflections</div></div>',unsafe_allow_html=True)
    ue=[e for e in st.session_state.entries if e["username"]==st.session_state.username]
    if not ue:
        st.info("No entries yet. Start a conversation in Daily Reflection to create your first entry.")
        return
    emots=sorted(set(e["primary"] for e in ue))
    filt=st.selectbox("Filter",["All"]+emots,label_visibility="collapsed")
    filtered=ue if filt=="All" else [e for e in ue if e["primary"]==filt]
    st.markdown(f'<div style="font-size:12px;color:#475569;margin-bottom:12px;">{len(filtered)} entries</div>',unsafe_allow_html=True)
    for e in filtered:
        tags="".join(f'<span class="emotion-tag">{em}</span>' for em in e["emotions"][:4])
        with st.expander(f"📅  {e['timestamp']}  ·  {e['primary'].capitalize()}"):
            st.markdown(f'<div style="font-size:13px;color:#cbd5e1;line-height:1.6;">{e["text"]}</div>',unsafe_allow_html=True)
            st.markdown(f'<div style="margin-top:8px;">{tags}</div>',unsafe_allow_html=True)
            st.markdown(f'<div class="reflection-box">💬 {TEMPLATES.get(e["primary"],TEMPLATES["neutral"])}</div>',unsafe_allow_html=True)

# ── ANALYTICS ─────────────────────────────────────────────────────────────────
def analytics_page():
    disc()
    st.markdown('<div class="page-header"><div class="header-title">📊 Analytics</div><div class="header-sub">Your emotional patterns at a glance</div></div>',unsafe_allow_html=True)
    ue=[e for e in st.session_state.entries if e["username"]==st.session_state.username]
    if len(ue)<2:
        st.info("Add at least 2 entries through the Daily Reflection page to see your analytics.")
        return
    df=pd.DataFrame(ue)
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Total Entries",len(ue))
    c2.metric("Unique Emotions",len(set(e["primary"] for e in ue)))
    top=df["primary"].value_counts().index[0]
    c3.metric("Top Emotion",top.capitalize())
    c4.metric("Session Entries",len(ue))
    st.markdown("---")
    cl,cr=st.columns(2)
    with cl:
        st.markdown("**Emotion Frequency**")
        all_emos=[]
        for e in ue: all_emos.extend(e["emotions"])
        top8=dict(sorted(Counter(all_emos).items(),key=lambda x:x[1],reverse=True)[:8])
        fig,ax=plt.subplots(figsize=(5,4))
        fig.patch.set_facecolor('#0a0f1e')
        ax.set_facecolor('#0a0f1e')
        ax.barh(list(top8.keys())[::-1],list(top8.values())[::-1],color='#6366f1',edgecolor='none',height=0.6)
        ax.tick_params(colors='#94a3b8',labelsize=9)
        for s in ax.spines.values(): s.set_color('#1e293b')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.grid(axis='x',color='#1e293b',linewidth=0.8)
        plt.tight_layout(); st.pyplot(fig); plt.close()
    with cr:
        st.markdown("**Primary Emotion Distribution**")
        pc=df["primary"].value_counts()
        colors=['#60a5fa','#a78bfa','#34d399','#f87171','#fbbf24','#818cf8','#f472b6','#2dd4bf']
        fig2,ax2=plt.subplots(figsize=(5,4))
        fig2.patch.set_facecolor('#0a0f1e'); ax2.set_facecolor('#0a0f1e')
        ax2.pie(pc.values,labels=pc.index,autopct='%1.0f%%',colors=colors[:len(pc)],textprops={'color':'#94a3b8','fontsize':9},pctdistance=0.82)
        plt.tight_layout(); st.pyplot(fig2); plt.close()
    if len(ue)>=3:
        st.markdown("---")
        st.markdown("**Emotion Timeline**")
        tl=pd.DataFrame([{"Entry":i+1,"Emotion":e["primary"].capitalize(),"Time":e["timestamp"]} for i,e in enumerate(reversed(ue))])
        st.dataframe(tl,use_container_width=True,hide_index=True)

def main():
    if not st.session_state.logged_in:
        login_page()
        return
    sidebar()
    p=st.session_state.page
    if p=="Chat": chat_page()
    elif p=="Entries": entries_page()
    elif p=="Analytics": analytics_page()

if __name__=="__main__":
    main()
