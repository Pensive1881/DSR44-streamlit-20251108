import streamlit as st
import pandas as pd
from plots import create_scatterplot
from model import load_model, get_feature_importance

# Task 1: Setup and Layout
st.set_page_config(layout="wide")

st.header("Worldwide Analysis of Quality of Life and Economic Factors")
st.write("This app enables you to explore the relationships between poverty, life expectancy, and GDP across various countries and years. Use the panels to select options and interact with the data.")

# Load data with caching
@st.cache_data
def load_data():
    DATA_URL = "https://raw.githubusercontent.com/JohannaViktor/streamlit_practical/refs/heads/main/global_development_data.csv"
    return pd.read_csv(DATA_URL)

df = load_data()

# Create 3 tabs
tab1, tab2, tab3 = st.tabs(["Global Overview", "Country Deep Dive", "Data Explorer"])

# Task 4 & 5 & 6: Tab 1 - Global Overview
with tab1:
    st.header("Global Overview")

    # Task 4: Year slider and key metrics
    year_min, year_max = int(df["year"].min()), int(df["year"].max())
    selected_year = st.slider("Select year", min_value=year_min, max_value=year_max, value=year_max, step=1)

    # Filter data by selected year
    df_year = df[df["year"] == selected_year]

    # Task 4: Create 4 key metrics in columns
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        mean_life_exp = df_year["Life Expectancy (IHME)"].mean()
        st.metric("Mean Life Expectancy", f"{mean_life_exp:.1f} years" if pd.notna(mean_life_exp) else "N/A")

    with col2:
        median_gdp = df_year["GDP per capita"].median()
        st.metric("Median GDP per Capita", f"${median_gdp:,.0f}" if pd.notna(median_gdp) else "N/A")

    with col3:
        mean_poverty = df_year["headcount_ratio_upper_mid_income_povline"].mean()
        st.metric("Mean Poverty Rate", f"{mean_poverty:.1f}%" if pd.notna(mean_poverty) else "N/A")

    with col4:
        num_countries = df_year["country"].nunique()
        st.metric("Number of Countries", f"{num_countries}")

    # Task 5: Scatterplot
    st.subheader("GDP vs Life Expectancy")
    fig = create_scatterplot(df_year)
    st.plotly_chart(fig, use_container_width=True)

    # Task 6: Model inference
    st.subheader("Life Expectancy Prediction")

    # Load the trained model
    model = load_model()

    if model is not None:
        # Input fields for inference
        col_input1, col_input2, col_input3 = st.columns(3)

        with col_input1:
            gdp_input = st.number_input(
                "GDP per capita ($)",
                min_value=float(df["GDP per capita"].min()),
                max_value=float(df["GDP per capita"].max()),
                value=float(df["GDP per capita"].median())
            )

        with col_input2:
            poverty_input = st.number_input(
                "Poverty Rate (%)",
                min_value=float(df["headcount_ratio_upper_mid_income_povline"].min()),
                max_value=float(df["headcount_ratio_upper_mid_income_povline"].max()),
                value=float(df["headcount_ratio_upper_mid_income_povline"].median())
            )

        with col_input3:
            year_input = st.number_input(
                "Year",
                min_value=int(df["year"].min()),
                max_value=int(df["year"].max()),
                value=int(df["year"].max())
            )

        # Make prediction
        if st.button("Predict Life Expectancy"):
            input_data = pd.DataFrame({
                'GDP per capita': [gdp_input],
                'headcount_ratio_upper_mid_income_povline': [poverty_input],
                'year': [year_input]
            })

            prediction = model.predict(input_data)[0]
            st.success(f"Predicted Life Expectancy: {prediction:.1f} years")

        # Feature importance
        st.subheader("Feature Importance")
        fig_importance = get_feature_importance(model)
        st.plotly_chart(fig_importance, use_container_width=True)
    else:
        st.warning("Model not trained yet. Please run model.py to train the model first.")

# Tab 2: Country Deep Dive
with tab2:
    st.header("Country Deep Dive")
    st.info("This tab is available for future development.")

# Task 2: Tab 3 - Data Explorer
with tab3:
    st.header("Data Explorer")

    # Controls
    countries = sorted(df["country"].dropna().unique().tolist())
    selected_countries = st.multiselect("Select countries", countries, default=countries[:5])

    year_range = st.slider(
        "Select year range",
        min_value=year_min,
        max_value=year_max,
        value=(year_min, year_max),
        step=1
    )

    # Filter data
    mask = df["year"].between(year_range[0], year_range[1])
    if selected_countries:
        mask &= df["country"].isin(selected_countries)
    filtered = df.loc[mask]

    # Show data
    st.dataframe(filtered, use_container_width=True)

    # Download button
    st.download_button(
        "Download filtered CSV",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="global_development_filtered.csv",
        mime="text/csv",
    )
