import plotly.express as px
import pandas as pd


def create_scatterplot(df):
    """
    Create a scatterplot of GDP per capita vs Life Expectancy

    Parameters:
    df (pd.DataFrame): DataFrame containing the data to plot

    Returns:
    plotly.graph_objects.Figure: The scatterplot figure
    """
    # Remove rows with missing values for the plot
    df_plot = df.dropna(subset=['GDP per capita', 'Life Expectancy (IHME)', 'country'])

    # Create the scatterplot
    fig = px.scatter(
        df_plot,
        x='GDP per capita',
        y='Life Expectancy (IHME)',
        color='headcount_ratio_upper_mid_income_povline',
        size='GDP per capita',
        hover_name='country',
        hover_data={
            'GDP per capita': ':,.0f',
            'Life Expectancy (IHME)': ':.1f',
            'headcount_ratio_upper_mid_income_povline': ':.1f',
            'year': True
        },
        log_x=True,
        title='GDP per Capita vs Life Expectancy',
        labels={
            'GDP per capita': 'GDP per Capita (USD, log scale)',
            'Life Expectancy (IHME)': 'Life Expectancy (years)',
            'headcount_ratio_upper_mid_income_povline': 'Poverty Rate (%)'
        },
        color_continuous_scale='RdYlGn_r'  # Red for high poverty, green for low poverty
    )

    # Update layout
    fig.update_layout(
        height=600,
        hovermode='closest'
    )

    return fig
