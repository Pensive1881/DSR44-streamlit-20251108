import pandas as pd
import pickle
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import plotly.graph_objects as go


MODEL_PATH = "life_expectancy_model.pkl"


def train_and_save_model():
    """
    Train a RandomForest model on the global development data
    and save it as a pickle file
    """
    print("Loading data...")
    DATA_URL = "https://raw.githubusercontent.com/JohannaViktor/streamlit_practical/refs/heads/main/global_development_data.csv"
    df = pd.read_csv(DATA_URL)

    # Select features and target
    features = ['GDP per capita', 'headcount_ratio_upper_mid_income_povline', 'year']
    target = 'Life Expectancy (IHME)'

    # Remove rows with missing values
    df_model = df[features + [target]].dropna()

    print(f"Training on {len(df_model)} samples...")

    # Split features and target
    X = df_model[features]
    y = df_model[target]

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train RandomForest model
    print("Training RandomForest model...")
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\nModel Performance:")
    print(f"R² Score: {r2:.4f}")
    print(f"RMSE: {mse**0.5:.4f}")

    # Save model as pickle
    print(f"\nSaving model to {MODEL_PATH}...")
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)

    print("Model saved successfully!")
    return model


def load_model():
    """
    Load the trained model from pickle file

    Returns:
    RandomForestRegressor or None: The trained model, or None if not found
    """
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        return model
    else:
        return None


def get_feature_importance(model):
    """
    Create a bar plot of feature importances

    Parameters:
    model: Trained RandomForest model

    Returns:
    plotly.graph_objects.Figure: Bar plot of feature importances
    """
    feature_names = ['GDP per capita', 'headcount_ratio_upper_mid_income_povline', 'year']
    importances = model.feature_importances_

    # Create DataFrame for plotting
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=True)

    # Create bar plot
    fig = go.Figure(go.Bar(
        x=importance_df['Importance'],
        y=importance_df['Feature'],
        orientation='h',
        marker_color='steelblue'
    ))

    fig.update_layout(
        title='Feature Importance for Life Expectancy Prediction',
        xaxis_title='Importance',
        yaxis_title='Feature',
        height=400
    )

    return fig


if __name__ == "__main__":
    # Train and save the model when this script is run directly
    train_and_save_model()
