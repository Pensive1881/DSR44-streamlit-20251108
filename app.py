import streamlit as st
import pandas as pd
#st.title("Hello, Streamlit!")
#st.write("Welcome to your first Streamlit app.")
#st.button("Click me!")

### TASK 1
st.set_page_config(layout="wide")

st.header("Worldwide Analysis of Quality of Life and Economic Factors")
st.write("This app enables you to explore the relationships between poverty,life expectancy, and GDP across various countries and years. Use the panels to select options and interact with the data.")

tab1, tab2, tab3 = st.tabs(["Global Overview", "Country Deep Dive", "Data Explorer"])

with tab1:
    st.header("Global Overview")
 
with tab2:
    st.header("Country Deep Dive")
   
with tab3:
    st.header("Data Explorer")

    ### TASK 2
    DATA_URL = "https://raw.githubusercontent.com/JohannaViktor/streamlit_practical/refs/heads/main/global_development_data.csv"
    df = pd.read_csv(DATA_URL)

    # controls
    countries = sorted(df["country"].dropna().unique().tolist())
    selected_countries = st.multiselect("Select countries", countries, default=countries[:5])

    year_min, year_max = int(df["year"].min()), int(df["year"].max())
    year_range = st.slider("Select year range", min_value=year_min, max_value=year_max, value=(year_min, year_max), step=1)

    # filter
    mask = df["year"].between(year_range[0], year_range[1])
    if selected_countries:
        mask &= df["country"].isin(selected_countries)
    filtered = df.loc[mask]

    # show + download
    st.dataframe(filtered, use_container_width=True)
    st.download_button(
        "Download filtered CSV",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="global_development_filtered.csv",
        mime="text/csv",
    )







