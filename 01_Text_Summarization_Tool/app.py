import streamlit as st
import arxiv
import pandas as pd
from transformers import pipeline

st.title("📝 Research Project with Summarization")

# Styling for input fields
st.markdown(
    """
    <style>
    .stTextInput>div>div>input {
        border: 2px solid #4CAF50; 
        border-radius: 5px; 
        padding: 8px;
        font-size: 16px;
    }
    </style>
    """, unsafe_allow_html=True
)

# Initialize session state for DataFrame and search query
if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame()
if "query" not in st.session_state:
    st.session_state.query = ""

# Input for query
query = st.text_input("🔍 Enter your research topic or query here:", value=st.session_state.query)

# Search button
btn = st.button("🔎 Search Papers")

# Function to fetch data
def getdata(query):
    try:
        # Search for papers
        search = arxiv.Search(
            query=query,
            max_results=10,
            sort_by=arxiv.SortCriterion.SubmittedDate
        )
        
        # Collect results
        papers = []
        for result in search.results():
            papers.append({
                "Published": result.published,
                "Title": result.title,
                "Abstract": result.summary,
                "Categories": ', '.join(result.categories)
            })
        
        # Create and display DataFrame
        if papers:
            st.session_state.df = pd.DataFrame(papers)  # Save DataFrame to session state
            pd.set_option('display.max_colwidth', None)
            st.dataframe(st.session_state.df)  # Display the DataFrame
        else:
            st.warning("No results found. Please try a different query.")
    
    except Exception as e:
        st.error(f"An error occurred: {e}")

# Trigger search on button click
if btn:
    if query.strip():  # Ensure the query is not empty
        st.session_state.query = query  # Save the query to session state
        getdata(query)
    else:
        st.warning("Please enter a query before searching.")

# Display search results if available
if not st.session_state.df.empty:
    st.dataframe(st.session_state.df)

# Input for abstract number
id = st.number_input("Enter the abstract number you wish to summarize (1-10):", min_value=1, max_value=10, step=1)

# Summarization button
btnA = st.button("📝 Summarize Abstract")

if btnA:
    if not st.session_state.df.empty:  # Ensure search results are available
        try:
            abstract = st.session_state.df.iloc[id - 1]['Abstract']  # Fetch abstract based on user input
            
            # Summarizer setup
            summarizer = pipeline("summarization", model="t5-small")
            chunk_size = 1000
            chunks = [abstract[i:i + chunk_size] for i in range(0, len(abstract), chunk_size)]
            
            # Summarize in chunks
            summarized_chunks = []
            for chunk in chunks:
                summary = summarizer(chunk, max_length=116, min_length=10, do_sample=False)
                summarized_chunks.append(summary[0]['summary_text'])
            
            # Combine summaries
            final_summary = " ".join(summarized_chunks)
            st.subheader("📃 Summarized Abstract:")
            st.write(final_summary)
        
        except Exception as e:
            st.error(f"An error occurred during summarization: {e}")
    else:
        st.warning("No search results available. Please perform a search first.")
