import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences


# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Movie Review Sentiment Analysis",
    page_icon="🎬",
    layout="wide"
)

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>

.stApp{
    background:
    linear-gradient(
    135deg,
    #d6f0ff,
    #fff7cc,
    #d9ffd9
    );
}

.main-title{
    text-align:center;
    font-size:48px;
    font-weight:bold;
    color:#003366;
    margin-bottom:5px;
}

.sub-title{
    text-align:center;
    font-size:22px;
    color:#444444;
    margin-bottom:30px;
}

.metric-card{
    background:white;
    padding:20px;
    border-radius:20px;
    box-shadow:0px 4px 12px rgba(0,0,0,0.15);
}

.stButton > button{
    width:100%;
    height:55px;
    border-radius:15px;
    border:none;
    font-size:18px;
    font-weight:bold;
    color:white;
    background:linear-gradient(
        135deg,
        #64c5eb,
        #8fd694
    );
}

.stTextArea textarea{
    border-radius:15px;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# LOAD MODELS
# =====================================================

@st.cache_resource
def load_models():

    rnn = load_model("simple_rnn_model.h5")

    lstm = load_model("lstm_model.h5")

    gru = load_model("gru_model.h5")

    return rnn, lstm, gru


model_rnn, model_lstm, model_gru = load_models()

# =====================================================
# IMDB WORD INDEX
# =====================================================

@st.cache_data
def load_word_index():
    return imdb.get_word_index()

word_index = load_word_index()

# =====================================================
# PREPROCESSING
# =====================================================

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

# =====================================================
# PREDICTION FUNCTION
# =====================================================

def predict_sentiment(review, model):

    review_encoded = encode_review(review)

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

    return sentiment, confidence, pred

# =====================================================
# HEADER
# =====================================================

st.markdown(
    '<div class="main-title">🎬 Movie Review Sentiment Analysis System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">Deep Learning Based Sentiment Classification</div>',
    unsafe_allow_html=True
)

# =====================================================
# MODEL SELECTION
# =====================================================

selected_model = st.selectbox(
    "Select Model",
    [
        "SimpleRNN",
        "LSTM",
        "GRU"
    ]
)

# =====================================================
# REVIEW INPUT
# =====================================================

review = st.text_area(
    "Enter your movie review here...",
    height=200
)

# =====================================================
# MODEL MAPPING
# =====================================================

model_map = {

    "SimpleRNN": model_rnn,

    "LSTM": model_lstm,

    "GRU": model_gru
}

# =====================================================
# ANALYZE BUTTON
# =====================================================

if st.button("Analyze Review"):

    if review.strip() == "":

        st.warning("Please enter a review.")

    else:

        model = model_map[selected_model]

        sentiment, confidence, pred = predict_sentiment(
            review,
            model
        )

        st.markdown("## Prediction Result")

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Sentiment",
                sentiment
            )

        with col2:

            st.metric(
                "Confidence",
                f"{confidence*100:.2f}%"
            )

        # ==========================================
        # PROBABILITY CHART
        # ==========================================

        prob_df = pd.DataFrame({

            "Class":[
                "Negative",
                "Positive"
            ],

            "Probability":[
                1-pred,
                pred
            ]
        })

        fig = px.bar(

            prob_df,

            x="Class",

            y="Probability",

            title="Confidence Chart"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

# =====================================================
# COMPARE ALL MODELS
# =====================================================

st.markdown("---")

st.header("Compare Predictions from All Models")

if st.button("Compare All Models"):

    if review.strip() == "":

        st.warning("Please enter a review.")

    else:

        results = []

        models = {

            "SimpleRNN": model_rnn,

            "LSTM": model_lstm,

            "GRU": model_gru
        }

        for name, model in models.items():

            sentiment, confidence, pred = predict_sentiment(
                review,
                model
            )

            results.append([

                name,

                sentiment,

                round(
                    confidence*100,
                    2
                )
            ])

        result_df = pd.DataFrame(

            results,

            columns=[

                "Model",

                "Predicted Sentiment",

                "Confidence (%)"
            ]
        )

        st.dataframe(
            result_df,
            use_container_width=True
        )

# =====================================================
# ACCURACY COMPARISON
# =====================================================

st.markdown("---")

st.header("Model Performance Dashboard")

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

    title="Accuracy Comparison"
)

st.plotly_chart(
    fig_acc,
    use_container_width=True
)

# =====================================================
# TRAINING TIME
# =====================================================

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

    title="Training Time Comparison (seconds)"
)

st.plotly_chart(
    fig_time,
    use_container_width=True
)

# =====================================================
# METRICS TABLE
# =====================================================

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

st.subheader("Performance Metrics")

st.dataframe(
    metrics_df,
    use_container_width=True
)