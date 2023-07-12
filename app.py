import streamlit as st
from code import preprocessing_data, graph_sentiment, analyse_mention, analyse_hastag, download_data, getAnalysisSVM

# Set page configuration
st.set_page_config(
    page_title="Twitter Sentiment Analysis 📊",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded",\
    menu_items={
         'About': "# TE mini Project under guidance of *Dr. LI sir* !"
         }
     
)

# Set background color
st.markdown(
    """
    <style>
    body {
        color: #555;
        background-color: #F5F5F5;
        
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Set title
st.title("Twitter Sentiment Analysis")

# Set functionality option
function_option = st.sidebar.selectbox("Select Functionality", ["Search By #Tag and Words", "Search By Username"])

sentiment_option = st.sidebar.selectbox("Select Sentiment Analysis Algorithm", ["TextBlob", "SVM"])

# Set search query
if function_option == "Search By #Tag and Words":
    word_query = st.text_input("Enter the #Hashtag or any word", "#")
else:
    word_query = st.text_input("Enter the Username (without @)")

# Set number of tweets to collect
number_of_tweets = st.slider("How many tweets do you want to collect for {}?".format(word_query), min_value=100, max_value=10000)

# Show information about expected wait time
st.info("1 tweet takes approx. 0.05 sec, so you may have to wait around {} minutes for {} tweets. Please have patience 🙌".format(round((number_of_tweets*0.05/60),2), number_of_tweets))


# Button to start sentiment analysis
if st.button("Analyze Sentiment"):
    if sentiment_option == "TextBlob":
        data = preprocessing_data(word_query, number_of_tweets, function_option)
        analyse = graph_sentiment(data)
        mention = analyse_mention(data)
        hashtag = analyse_hastag(data)

        # Show extracted and preprocessed dataset
        st.header("Extracted and Preprocessed Dataset")
        st.write(data)
        download_data(data, label="twitter_sentiment_filtered")

        # Show EDA on the data
        col1, col2, col3 = st.columns(3)
        with col2:
            st.markdown("### Exploratory Data Analysis")

        col1, col2 = st.columns(2)
        with col1:
            st.text("Top 10 @Mentions in {} tweets".format(number_of_tweets))
            st.bar_chart(mention)

        with col2:
            st.text("Top 10 Hashtags used in {} tweets".format(number_of_tweets))
            st.bar_chart(hashtag)

        col3, col4 = st.columns(2)
        with col3:
            st.text("Top 10 Used Links for {} tweets".format(number_of_tweets))
            st.bar_chart(data["links"].value_counts().head(10).reset_index())

        with col4:
            st.text("All the Tweets that contain top 10 links used")
            filtered_data = data[data["links"].isin(data["links"].value_counts().head(10).reset_index()["index"].values)]
            st.write(filtered_data)


        # Show Twitter sentiment analysis
        st.header("Twitter Sentiment Analysis")
        st.bar_chart(analyse)
    
    elif sentiment_option == "SVM":
        # text_query = st.text_input("Enter text to analyze sentiment")
        analysis = getAnalysisSVM('text', 'svm_model', 'vectorizer')
        if analysis == 0:
            st.write("The text is Negative")
        elif analysis == 1:
            st.write("The text is Neutral")
        elif analysis == 2:
            st.write("The text is Positive")

