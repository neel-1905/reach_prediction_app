from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

app = Flask(__name__)

# Initialize global variables for model components
tfidf = None
scaler = None
model = None
numeric_features_count = 6
mse = None
rmse = None
r2_score = None

def plot_to_base64(plt):
    """Convert a matplotlib plot to a base64 encoded image."""
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode('utf-8')

@app.route('/', methods=['GET', 'POST'])
def index():
    global tfidf, scaler, model, mse, rmse, r2_score, numeric_features_count

    if request.method == 'POST':
        if 'file' in request.files:
            file = request.files['file']

            if file.filename == '':
                return {"error": "No file selected."}, 400

            try:
                # Load the dataset from the uploaded file
                data = pd.read_csv(file, encoding='latin1')

                # Feature Engineering
                numeric_features = data[['Likes', 'Saves', 'Comments', 'Shares', 'Profile Visits', 'Follows']].values

                # Process Hashtags: TF-IDF Vectorization
                tfidf = TfidfVectorizer(max_features=50)
                hashtag_features = tfidf.fit_transform(data['Hashtags']).toarray()

                # Combine Numeric and Hashtag Features
                X = np.hstack([numeric_features, hashtag_features])
                y = data['Impressions'].values

                # Train-Test Split
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Normalize Numeric Features (First 6 columns are numeric)
                scaler = StandardScaler()
                X_train[:, :numeric_features_count] = scaler.fit_transform(X_train[:, :numeric_features_count])
                X_test[:, :numeric_features_count] = scaler.transform(X_test[:, :numeric_features_count])

                # Train a PassiveAggressiveRegressor
                model = PassiveAggressiveRegressor(max_iter=1000, random_state=42, tol=1e-3)
                model.fit(X_train, y_train)

                # Evaluate the model
                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2_score = model.score(X_test, y_test)

                # Analysis 1: Feature Importance
                feature_importance = model.coef_[:numeric_features_count]
                plt.figure(figsize=(10, 5))
                sns.barplot(x=['Likes', 'Saves', 'Comments', 'Shares', 'Profile Visits', 'Follows'], y=feature_importance)
                plt.title('Feature Importance (Numeric Features)')
                feature_importance_plot = plot_to_base64(plt)

                # Analysis 2: Hashtag Frequency
                hashtag_freq = pd.Series(' '.join(data['Hashtags']).split()).value_counts().head(10)
                plt.figure(figsize=(10, 5))
                sns.barplot(x=hashtag_freq.index, y=hashtag_freq.values)
                plt.title('Top 10 Hashtags by Frequency')
                hashtag_freq_plot = plot_to_base64(plt)

                # Analysis 3: Impressions Distribution
                plt.figure(figsize=(10, 5))
                sns.histplot(data['Impressions'], kde=True)
                plt.title('Distribution of Impressions')
                impressions_dist_plot = plot_to_base64(plt)

                # Analysis 4: Correlation Heatmap
                plt.figure(figsize=(10, 5))
                sns.heatmap(data[['Likes', 'Saves', 'Comments', 'Shares', 'Profile Visits', 'Follows', 'Impressions']].corr(), annot=True, cmap='coolwarm')
                plt.title('Correlation Heatmap')
                correlation_heatmap = plot_to_base64(plt)

                # Analysis 5: Predicted vs Actual Impressions
                plt.figure(figsize=(10, 5))
                sns.scatterplot(x=y_test, y=y_pred)
                plt.xlabel('Actual Impressions')
                plt.ylabel('Predicted Impressions')
                plt.title('Predicted vs Actual Impressions')
                predicted_vs_actual_plot = plot_to_base64(plt)

                return {
                    "mse": f"{mse:.2f}",
                    "rmse": f"{rmse:.2f}",
                    "r2_score": f"{r2_score:.4f}",
                    "feature_importance_plot": feature_importance_plot,
                    "hashtag_freq_plot": hashtag_freq_plot,
                    "impressions_dist_plot": impressions_dist_plot,
                    "correlation_heatmap": correlation_heatmap,
                    "predicted_vs_actual_plot": predicted_vs_actual_plot
                }

            except Exception as e:
                return {"error": str(e)}, 500

        return {"error": "No file uploaded."}, 400

    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global tfidf, scaler, model, numeric_features_count

    try:
        input_data = request.form

        # Process numeric inputs
        numeric_inputs = np.array([[
            float(input_data['Likes']),
            float(input_data['Saves']),
            float(input_data['Comments']),
            float(input_data['Shares']),
            float(input_data['Profile Visits']),
            float(input_data['Follows'])
        ]])
        numeric_inputs = scaler.transform(numeric_inputs)

        # Process hashtags
        hashtag_inputs = tfidf.transform([input_data['Hashtags']]).toarray()

        # Combine numeric and hashtag inputs
        input_features = np.hstack([numeric_inputs, hashtag_inputs])

        # Predict impressions
        predicted_reach = model.predict(input_features)[0]

        return {"Predicted Reach (Impressions)": f"{predicted_reach:.2f}"}

    except Exception as e:
        return {"error": str(e)}, 500

if __name__ == '__main__':
    app.run(debug=True)