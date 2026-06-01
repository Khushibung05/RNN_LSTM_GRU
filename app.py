import streamlit as st
import numpy as np
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go

from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Movie Sentiment AI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==========================================================
# CYBERPUNK CSS
# ==========================================================

st.markdown("""
<style>

/* ==========================================================
BACKGROUND
========================================================== */

.stApp{

background:
linear-gradient(
135deg,
#050816 0%,
#0b1120 35%,
#111827 70%,
#050816 100%
);

color:white;
}


/* ==========================================================
REMOVE STREAMLIT MENU
========================================================== */

#MainMenu{
visibility:hidden;
}

footer{
visibility:hidden;
}

header{
visibility:hidden;
}


/* ==========================================================
HEADER
========================================================== */

.main-title{

font-size:60px;

font-weight:900;

text-align:center;

margin-top:10px;

color:#00e5ff;

text-shadow:

0 0 10px #00e5ff,

0 0 25px #00e5ff,

0 0 50px #00e5ff;
}


.sub-title{

text-align:center;

font-size:24px;

color:#94a3b8;

margin-bottom:40px;
}


/* ==========================================================
GLASS CARD
========================================================== */

.glass{

background:
rgba(255,255,255,0.05);

backdrop-filter:
blur(15px);

padding:25px;

border-radius:20px;

border:
1px solid rgba(255,255,255,0.1);

box-shadow:
0 0 20px rgba(0,229,255,0.15);

margin-bottom:20px;
}


/* ==========================================================
SYSTEM CARD
========================================================== */

.status-card{

background:
rgba(0,229,255,0.08);

padding:20px;

border-radius:20px;

border:
1px solid rgba(0,229,255,0.2);

box-shadow:
0 0 25px rgba(0,229,255,0.2);
}


/* ==========================================================
MODEL CARD
========================================================== */

.model-card{

background:
rgba(255,255,255,0.04);

padding:20px;

border-radius:20px;

text-align:center;

border:
1px solid rgba(255,255,255,0.1);

box-shadow:
0 0 15px rgba(0,229,255,0.15);
}
.model-card:hover{

transform:translateY(-8px);

transition:0.3s;

box-shadow:
0 0 30px #00e5ff;
}

/* ==========================================================
BUTTON
========================================================== */

.stButton > button{

width:100%;

height:55px;

font-size:18px;

font-weight:bold;

border:none;

border-radius:15px;

background:
linear-gradient(
90deg,
#00e5ff,
#2563eb
);

color:white;

box-shadow:
0 0 15px #00e5ff;
}


/* ==========================================================
TEXTAREA
========================================================== */

.stTextArea textarea{

background:#0f172a !important;

color:white !important;

border:
1px solid #00e5ff !important;

border-radius:15px;
}


/* ==========================================================
SELECT BOX
========================================================== */

div[data-baseweb="select"]{

background:#0f172a;
}


/* ==========================================================
METRICS
========================================================== */

[data-testid="metric-container"]{

background:
rgba(255,255,255,0.04);

border:
1px solid rgba(255,255,255,0.1);

padding:15px;

border-radius:15px;

box-shadow:
0 0 15px rgba(0,229,255,0.1);
}


/* ==========================================================
TABLES
========================================================== */

[data-testid="stDataFrame"]{

border-radius:15px;
}


/* ==========================================================
TABS
========================================================== */

.stTabs [data-baseweb="tab"]{

font-size:18px;
font-weight:bold;
}

</style>
""", unsafe_allow_html=True)

# ==========================================================
# LOAD MODELS
# ==========================================================

@st.cache_resource
def load_models():

    model_rnn = load_model(
        "simple_rnn_model.h5",
        compile=False
    )

    model_lstm = load_model(
        "lstm_model.h5",
        compile=False
    )

    model_gru = load_model(
        "gru_model.h5",
        compile=False
    )

    return (
        model_rnn,
        model_lstm,
        model_gru
    )


model_rnn, model_lstm, model_gru = load_models()

# ==========================================================
# LOAD IMDB WORD INDEX
# ==========================================================

@st.cache_data
def load_word_index():

    return imdb.get_word_index()

word_index = load_word_index()

# ==========================================================
# REVIEW ENCODER
# ==========================================================

def encode_review(review):

    words = review.lower().split()

    encoded = []

    for word in words:

        if word in word_index:

            encoded.append(
                word_index[word] + 3
            )

    padded = pad_sequences(
        [encoded],
        maxlen=500
    )

    return padded

# ==========================================================
# PREDICTION FUNCTION
# ==========================================================

def predict_sentiment(review, model):

    review_encoded = encode_review(
        review
    )

    pred = model.predict(
        review_encoded,
        verbose=0
    )[0][0]

    sentiment = (
        "Positive"
        if pred >= 0.5
        else "Negative"
    )

    confidence = (
        pred
        if pred >= 0.5
        else 1 - pred
    )

    return (
        sentiment,
        confidence,
        pred
    )
# ==========================================================
# HERO HEADER
# ==========================================================

st.markdown(
"""
<div class="main-title">
🎬 MOVIE SENTIMENT AI
</div>
""",
unsafe_allow_html=True
)

st.markdown(
"""
<div class="sub-title">
Deep Learning Powered Sentiment Intelligence Console
</div>
""",
unsafe_allow_html=True
)

# ==========================================================
# TOP DASHBOARD LAYOUT
# ==========================================================

left_panel, right_panel = st.columns(
    [2,1]
)

# ==========================================================
# LEFT PANEL
# ==========================================================

with left_panel:

    st.markdown(
    """
    <div class="glass">
    <h3>📝 Review Analysis Console</h3>
    </div>
    """,
    unsafe_allow_html=True
    )

    selected_model = st.selectbox(

        "Select Prediction Model",

        [
            "SimpleRNN",
            "LSTM",
            "GRU"
        ]
    )

    review = st.text_area(

        "Enter your movie review here...",

        height=260,

        placeholder="""
Example:

This movie was absolutely fantastic.
The acting was brilliant and the story was engaging.
        """
    )

    analyze_btn = st.button(
        "🚀 Analyze Review"
    )

# ==========================================================
# RIGHT PANEL
# ==========================================================

with right_panel:

    st.markdown(
    """
    <div class="status-card">

    <h3>🟢 AI SYSTEM STATUS</h3>

    <hr>

    ✔ SimpleRNN Loaded

    <br>

    ✔ LSTM Loaded

    <br>

    ✔ GRU Loaded

    <br><br>

    ⚡ GPU Ready

    <br>

    🧠 NLP Engine Active

    <br>

    🔒 Secure Prediction Mode

    </div>
    """,
    unsafe_allow_html=True
    )

# ==========================================================
# MODEL MAP
# ==========================================================

model_map = {

    "SimpleRNN": model_rnn,

    "LSTM": model_lstm,

    "GRU": model_gru
}

# ==========================================================
# ANALYSIS SECTION
# ==========================================================

if analyze_btn:

    if review.strip() == "":

        st.warning(
            "Please enter a review."
        )

    else:

        model = model_map[
            selected_model
        ]

        sentiment, confidence, pred = predict_sentiment(
            review,
            model
        )

        # ==================================================
        # RESULT ROW
        # ==================================================

        result_col, gauge_col = st.columns(
            [1,1]
        )

        # ==============================================
        # RESULT CARD
        # ==============================================

        with result_col:

            st.markdown(
            """
            <div class="glass">
            <h3>🎯 Prediction Result</h3>
            </div>
            """,
            unsafe_allow_html=True
            )

            st.metric(

                label="Predicted Sentiment",

                value=sentiment
            )

            st.metric(

                label="Confidence Score",

                value=f"{confidence*100:.2f}%"
            )

        # ==============================================
        # CYBERPUNK GAUGE
        # ==============================================

        with gauge_col:

            gauge = go.Figure(

                go.Indicator(

                    mode="gauge+number",

                    value=confidence*100,

                    title={
                        'text':
                        "AI Confidence"
                    },

                    gauge={

                        'axis':{
                            'range':[0,100]
                        },

                        'bar':{
                            'color':'#00e5ff'
                        },

                        'bgcolor':
                        '#111827',

                        'borderwidth':2,

                        'bordercolor':
                        '#00e5ff'
                    }
                )
            )

            gauge.update_layout(

                paper_bgcolor=
                "#0b1120",

                font_color=
                "white",

                height=350
            )

            st.plotly_chart(

                gauge,

                use_container_width=True
            )

        # ==================================================
        # RADAR CHART
        # ==================================================

        st.markdown("## 🕸 Sentiment Radar")

        radar = go.Figure()

        radar.add_trace(

            go.Scatterpolar(

                r=[
                    pred,
                    1-pred
                ],

                theta=[
                    "Positive",
                    "Negative"
                ],

                fill='toself',

                line=dict(

                    color=
                    "#00e5ff",

                    width=4
                )
            )
        )

        radar.update_layout(

            paper_bgcolor=
            "#0b1120",

            plot_bgcolor=
            "#0b1120",

            font_color=
            "white",

            polar=dict(

                bgcolor=
                "#0b1120",

                radialaxis=dict(

                    visible=True,

                    range=[0,1]
                )
            ),

            height=500
        )

        st.plotly_chart(

            radar,

            use_container_width=True
        )
    # ==========================================================
# MODEL BATTLE ARENA
# ==========================================================

st.markdown("---")

st.markdown(
"""
<div class="glass">
<h2>🤖 AI MODEL BATTLE ARENA</h2>
</div>
""",
unsafe_allow_html=True
)

compare_btn = st.button(
    "⚔ Compare All Models"
)

if compare_btn:

    if review.strip() == "":

        st.warning(
            "Please enter a review first."
        )

    else:

        models = {

            "SimpleRNN": model_rnn,

            "LSTM": model_lstm,

            "GRU": model_gru
        }

        card1, card2, card3 = st.columns(3)

        columns = [
            card1,
            card2,
            card3
        ]

        for col, (name, model) in zip(
            columns,
            models.items()
        ):

            sentiment, confidence, pred = predict_sentiment(
                review,
                model
            )

            with col:

                st.markdown(f"""
                <div class="model-card">

                <h2>{name}</h2>

                <hr>

                <h3>{sentiment}</h3>

                <h1 style='color:#00e5ff'>
                {confidence*100:.2f}%
                </h1>

                </div>
                """,
                unsafe_allow_html=True)

# ==========================================================
# PERFORMANCE DASHBOARD
# ==========================================================

st.markdown("---")

st.markdown(
"""
<div class="glass">
<h2>📊 PERFORMANCE ANALYTICS</h2>
</div>
""",
unsafe_allow_html=True
)

tab1, tab2, tab3, tab4 = st.tabs([

    "📈 Accuracy",

    "⚡ Training Time",

    "📊 Metrics",

    "🏆 Ranking"
])

# ==========================================================
# TAB 1 - ACCURACY
# ==========================================================

with tab1:

    acc_df = pd.DataFrame({

        "Model":[

            "SimpleRNN",

            "LSTM",

            "GRU"
        ],

        "Accuracy":[

            0.50536,

            0.51416,

            0.50624
        ]
    })

    fig_acc = px.bar(

        acc_df,

        x="Model",

        y="Accuracy",

        color="Model",

        title=
        "Model Accuracy Comparison"
    )

    fig_acc.update_layout(

        paper_bgcolor=
        "#0b1120",

        plot_bgcolor=
        "#0b1120",

        font_color=
        "white"
    )

    st.plotly_chart(

        fig_acc,

        use_container_width=True
    )

# ==========================================================
# TAB 2 - TRAINING TIME
# ==========================================================

with tab2:

    time_df = pd.DataFrame({

        "Model":[

            "SimpleRNN",

            "LSTM",

            "GRU"
        ],

        "Training Time":[

            72.83,

            42.59,

            43.91
        ]
    })

    fig_time = px.bar(

        time_df,

        x="Model",

        y="Training Time",

        color="Model",

        title=
        "Training Time Comparison"
    )

    fig_time.update_layout(

        paper_bgcolor=
        "#0b1120",

        plot_bgcolor=
        "#0b1120",

        font_color=
        "white"
    )

    st.plotly_chart(

        fig_time,

        use_container_width=True
    )

# ==========================================================
# TAB 3 - METRICS
# ==========================================================

with tab3:

    metrics_df = pd.DataFrame({

        "Metric":[

            "Accuracy",

            "Precision",

            "Recall",

            "F1 Score"
        ],

        "SimpleRNN":[

            0.50536,

            0.55805,

            0.05152,

            0.09433
        ],

        "LSTM":[

            0.51416,

            0.67052,

            0.05568,

            0.10282
        ],

        "GRU":[

            0.50624,

            0.58159,

            0.04448,

            0.08264
        ]
    })

    st.dataframe(

        metrics_df,

        use_container_width=True
    )

# ==========================================================
# TAB 4 - MODEL RANKING
# ==========================================================

with tab4:

    st.markdown(
    """
    <div class="glass">

    <h2>👑 MODEL LEADERBOARD</h2>

    <hr>

    🥇 LSTM — Accuracy: 51.42%

    <br><br>

    🥈 GRU — Accuracy: 50.62%

    <br><br>

    🥉 SimpleRNN — Accuracy: 50.54%

    </div>
    """,
    unsafe_allow_html=True
    )

# ==========================================================
# FOOTER
# ==========================================================

st.markdown("---")

st.markdown(
"""
<div style='text-align:center;
color:#94a3b8;
padding:20px;'>

🚀 Movie Sentiment AI Console

<br>

Powered by Deep Learning

<br>

SimpleRNN • LSTM • GRU

</div>
""",
unsafe_allow_html=True
)