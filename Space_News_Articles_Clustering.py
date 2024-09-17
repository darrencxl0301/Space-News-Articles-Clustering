import streamlit as st
import os
import streamlit as st
from bokeh.plotting import figure
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Slider, TextInput, Div, HoverTool
from bokeh.palettes import Category20
from bokeh.transform import factor_cmap
import numpy as np
import pandas as pd
import pickle
from sklearn.decomposition import PCA
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering, SpectralClustering, Birch
import base64

def autoplay_audio(file_path: str):
    with open(file_path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        md = f"""
            <audio id="themeAudio" autoplay="true" style="display:none;">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            """
        st.markdown(md, unsafe_allow_html=True)


# Page selection for navigation
page = st.sidebar.selectbox("Select a page:", ["Home", "Dashboard"])

if page == "Home":
    
    # Load space PNG image and display it
    space_image_url = 'space_image.png'
    b64 = None
    with open(space_image_url, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()

    st.image(space_image_url, caption="Exploring Space News", use_column_width=True)
    style = f"""
            <style>
            .stApp {{
                background: url(data:image/jpg;base64,{b64});
                background-size: cover;
            }}
            h1, h2, p, [data-testid="stImageCaption"]{{
                color: white;
            }}
            </style>
            """
    st.markdown(style, unsafe_allow_html=True)
    
    # Big title for the app
    st.title("Space News Articles Clustering")


    # Add autoplay audio
    audio_url = 'starwars.mp3'
    autoplay_audio(audio_url)
    
    # Subtitle for the description
    st.markdown("## Description")

    # Updated paragraph
    st.markdown("""
    The ensemble model combines **GMM** with 12 components and **t-SNE** for dimensionality reduction, allowing us to visualize high-dimensional data in a clear and interpretable way.
    """)

    # Load topics from 'topics.txt' file using os.path.join
    topic_path = os.path.join(os.getcwd(), 'topics.txt')
    with open(topic_path) as f:
        topics = [line.strip() for line in f.readlines()]  # Each line corresponds to keywords for a specific cluster

    # Associate each cluster (0 to 12) with its corresponding keywords from 'topics.txt'
    cluster_keywords = {
        i: topics[i] if i < len(topics) else "Not enough instances to be determined." for i in range(13)
    }

    # Load X_embedded
    with open('X_embedded.pkl', 'rb') as f:
        X_embedded = pickle.load(f)

    # Load y_labels
    with open('y_labels.pkl', 'rb') as f:
        y_labels = pickle.load(f)

    # Load filtered_df
    filtered_df = pd.read_pickle('filtered_df.pkl')

    # Data source for Bokeh
    def filter_data(cluster_value, search_term):
        """Filter the data based on the cluster value and search term."""
        if cluster_value == 12:  # Show all clusters
            mask = filtered_df['title'].str.contains(search_term, case=False, na=False)
        else:
            mask = (y_labels == cluster_value) & filtered_df['title'].str.contains(search_term, case=False, na=False)

        return X_embedded[mask], filtered_df[mask], y_labels[mask]

    # Create Bokeh plot
    hover = HoverTool(tooltips=[("Title", "@titles{safe}"), ("Date", "@date{safe}")], point_policy="follow_mouse")

    def create_plot(x_data, titles, dates, labels):
        source = ColumnDataSource(data=dict(
            x=x_data[:, 0],
            y=x_data[:, 1],
            titles=titles,
            date=dates,
            desc=[str(label) for label in labels]  # Convert labels to strings for color mapping
        ))

        # Use Category20 palette for 12 clusters (using Category20[12])
        palette = Category20[12]  # Adjusted palette with 12 distinct colors

        # Use factor_cmap to map cluster labels to colors
        mapper = factor_cmap('desc', palette=palette, factors=[str(i) for i in range(12)])  # Assuming clusters 0 to 11

        plot = figure(width=700, height=500, tools=[hover, 'pan', 'wheel_zoom', 'box_zoom', 'reset', 'tap'],
                    title="Clustering of the space news articles with t-SNE and GMM", toolbar_location="above")

        plot.scatter('x', 'y', size=5, source=source, fill_color=mapper, line_alpha=0.3, line_color="black", legend_field='desc')

        return plot

    # Add Widgets for filtering
    slider = st.slider("Cluster #", 0, 12, 12, help="Slide to filter clusters, 12 means all clusters")
    keyword = st.text_input("Search:", "", help="Search by keywords")

    # Display the keywords for the selected cluster in a smaller font using HTML
    if 0 <= slider <= 11:
        st.markdown(f"<b>Keywords for Cluster {slider}</b>", unsafe_allow_html=True)
    else:
        st.markdown("<b>All Clusters</b>", unsafe_allow_html=True)
    st.markdown(f"<small>{cluster_keywords.get(slider, 'No keywords available for this cluster.')}</small>", unsafe_allow_html=True)


    # Filter data based on slider and keyword input
    x_data, filtered_titles, filtered_labels = filter_data(slider, keyword)

    # Create updated plot
    plot = create_plot(x_data, filtered_titles['title'], filtered_titles['date'], filtered_labels)

    # Display the plot in Streamlit
    st.bokeh_chart(plot)

elif page == "Dashboard":
    st.title("Interactive Clustering Dashboard")

    with open('X_embedded.pkl', 'rb') as f:
        X_embedded = pickle.load(f)

    # Load y_labels for clustering labels (optional if used)
    with open('y_labels.pkl', 'rb') as f:
        y_labels = pickle.load(f)

    # Data Input: File upload or text input for user-uploaded data
    uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

    if uploaded_file is not None:
        # Load the data
        df = pd.read_csv(uploaded_file)
        st.write("Data Preview:", df.head())

        # Use X_embedded for dimensionality reduction instead of full dataset
        X = X_embedded

        # Data Exploration: Filter and sort with selectbox and sliders
        # Data Exploration: Filter and sort with selectbox and sliders
        if st.checkbox("Explore Data"):
            column_to_filter = st.selectbox("Select a column to filter:", df.columns)

            # Check if the selected column is of datetime type or can be converted to datetime
            if pd.api.types.is_datetime64_any_dtype(df[column_to_filter]):
                # Handle date columns with st.date_input
                st.write("Date column detected. Please select a date range.")
                start_date = st.date_input(f"Start date for {column_to_filter}:", value=pd.to_datetime(df[column_to_filter]).min())
                end_date = st.date_input(f"End date for {column_to_filter}:", value=pd.to_datetime(df[column_to_filter]).max())
                filtered_data = df[(pd.to_datetime(df[column_to_filter]) >= start_date) & (pd.to_datetime(df[column_to_filter]) <= end_date)]

            elif pd.api.types.is_numeric_dtype(df[column_to_filter]):
                # Handle numeric columns with st.slider
                filter_value = st.slider(f"Select a value for {column_to_filter}:", min_value=int(df[column_to_filter].min()), max_value=int(df[column_to_filter].max()))
                filtered_data = df[df[column_to_filter] == filter_value]

            else:
                # Handle non-numeric, non-date columns with st.selectbox
                filter_value = st.selectbox(f"Select a value for {column_to_filter}:", df[column_to_filter].unique())
                filtered_data = df[df[column_to_filter] == filter_value]

            st.write("Filtered Data", filtered_data)


        # PCA Visualization: Apply PCA and visualize the data in reduced dimensions
        # PCA Visualization: Apply PCA and visualize the data in reduced dimensions
        # PCA Visualization: Apply PCA and visualize the data in reduced dimensions
        if st.checkbox("PCA Visualization"):
            # PCA with 2 components
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(df.select_dtypes(include=[np.number]))

            # Display explained variance ratio for each component
            st.write(f"PCA Explained Variance Ratio for PC1 and PC2: {pca.explained_variance_ratio_}")

            # Create a DataFrame with PCA results
            pca_df = pd.DataFrame(X_embedded, columns=['PC1', 'PC2'])
            pca_df['Cluster'] = df['Cluster'] if 'Cluster' in df.columns else np.random.randint(0, 12, len(X_embedded))

            # Plot 2D scatter plot for PC1 vs PC2
            pca_fig = px.scatter(pca_df, x='PC1', y='PC2', color='Cluster', title="PCA 2D Cluster Visualization", color_continuous_scale='Viridis')
            st.plotly_chart(pca_fig)

            # 3D PCA Visualization
            pca_3d = PCA(n_components=3)
            pca_3d_result = pca_3d.fit_transform(df.select_dtypes(include=[np.number]))
            pca_3d_df = pd.DataFrame(pca_3d_result, columns=['PC1', 'PC2', 'PC3'])
            pca_3d_df['Cluster'] = df['Cluster'] if 'Cluster' in df.columns else np.random.randint(0, 12, len(df))

            # 3D scatter plot for PC1, PC2, and PC3
            pca_3d_fig = px.scatter_3d(pca_3d_df, x='PC1', y='PC2', z='PC3', color='Cluster', title="PCA 3D Cluster Visualization", color_continuous_scale='Plasma')
            st.plotly_chart(pca_3d_fig)

            # Visualize loadings: how features contribute to each principal component
            st.markdown("### Feature Contribution to Principal Components")
            loadings = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2'], index=df.select_dtypes(include=[np.number]).columns)
            st.write(loadings)

            # Bar plot to show the contribution of features to PC1 and PC2
            loadings_fig = px.bar(loadings, x=loadings.index, y='PC1', title="Feature Contribution to PC1")
            st.plotly_chart(loadings_fig)

            loadings_fig2 = px.bar(loadings, x=loadings.index, y='PC2', title="Feature Contribution to PC2")
            st.plotly_chart(loadings_fig2)



        # Interactive Charts: Dynamic visualizations with Plotly or Altair
        # Interactive Charts: Dynamic visualizations with meaningful colors
        # Interactive Charts: Dynamic visualizations with meaningful colors
        if st.checkbox("Interactive Charts"):
            chart_type = st.selectbox("Select Chart Type", ["2D Scatter", "2D Line Plot", "3D Scatter"])
            color_by = st.selectbox("Color by", options=df.columns)  # Choose column to color by

            if chart_type == "2D Scatter":
                # Plot a 2D scatter plot
                fig = px.scatter(df, x=df.columns[1], y=df.columns[2], color=color_by, title="2D Scatter Plot")
                st.plotly_chart(fig)

            elif chart_type == "2D Line Plot":
                # Plot a 2D line plot
                fig = px.line(df, x=df.columns[1], y=df.columns[2], color=color_by, title="2D Line Plot")
                st.plotly_chart(fig)

            elif chart_type == "3D Scatter":
                # Plot a 3D scatter plot
                fig = px.scatter_3d(df, x=df.columns[1], y=df.columns[2], z=df.columns[3], color=color_by, title="3D Scatter Plot")
                st.plotly_chart(fig)



        # Model Performance Metrics: Display clustering performance metrics

        # Model Performance Metrics and Dynamic Parameter Adjustment
        if st.checkbox("Model Performance Metrics and Clustering"):
            model_type = st.selectbox("Select model for metrics and clustering:", ['GMM', 'Agglomerative Clustering', 'Spectral Clustering', 'Birch Clustering'])

            # Input parameters based on model type
            if model_type == 'GMM':
                n_components = st.slider("Number of Components (Clusters)", min_value=2, max_value=10, value=3)
                covariance_type = st.selectbox("Covariance Type", ['full', 'tied', 'diag', 'spherical'])
                model = GaussianMixture(n_components=n_components, covariance_type=covariance_type)
                predicted_labels = model.fit_predict(X_embedded)

            elif model_type == 'Agglomerative Clustering':
                n_clusters = st.slider("Number of Clusters", min_value=2, max_value=10, value=3)
                linkage = st.selectbox("Linkage Type", ['ward', 'complete', 'average', 'single'])
                model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
                predicted_labels = model.fit_predict(X_embedded)

            elif model_type == 'Spectral Clustering':
                n_clusters = st.slider("Number of Clusters", min_value=2, max_value=10, value=3)
                affinity = st.selectbox("Affinity", ['nearest_neighbors', 'rbf'])
                if affinity == 'nearest_neighbors':
                    n_neighbors = st.slider("Number of Neighbors", min_value=2, max_value=25, value=10)  # New slider for n_neighbors
                    model = SpectralClustering(n_clusters=n_clusters, affinity=affinity, n_neighbors=n_neighbors, assign_labels='kmeans')
                else:
                    model = SpectralClustering(n_clusters=n_clusters, affinity=affinity, assign_labels='kmeans')
                predicted_labels = model.fit_predict(X_embedded)

            elif model_type == 'Birch Clustering':
                n_clusters = st.slider("Number of Clusters", min_value=2, max_value=10, value=3)
                threshold = st.slider("Threshold", min_value=0.1, max_value=1.0, value=0.5)
                model = Birch(n_clusters=n_clusters, threshold=threshold)
                predicted_labels = model.fit_predict(X_embedded)

            # Calculate clustering metrics
            silhouette_avg = silhouette_score(X_embedded, predicted_labels)
            davies_bouldin = davies_bouldin_score(X_embedded, predicted_labels)
            calinski_harabasz = calinski_harabasz_score(X_embedded, predicted_labels)

            # Create two columns: one for metrics and one for the cluster plot
            col1, col2 = st.columns([1, 3])  # Left column (1) for metrics, right column (3) for plot

            # Display metrics on the left side in a more compact form
            with col1:
                st.caption("Performance Metrics:")
                st.metric("Silhouette", round(silhouette_avg, 2), delta=None, help="Silhouette Score (higher is better)")
                st.metric("Davies-Bouldin", round(davies_bouldin, 2), delta=None, help="Davies-Bouldin Index (lower is better)")
                st.metric("Calinski-Harabasz", round(calinski_harabasz, 2), delta=None, help="Calinski-Harabasz Index (higher is better)")

            # Plot the clustering result on the right side
            with col2:
                fig = go.Figure()

                custom_colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']

                fig.add_trace(go.Scatter(
                    x=X_embedded[:, 0],
                    y=X_embedded[:, 1],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=predicted_labels,
                        colorscale=[(i / (len(custom_colors) - 1), color) for i, color in enumerate(custom_colors)],  # Custom color scale
                        showscale=True
                    ),
                    name=f'{model_type} Clusters'
                ))

                fig.update_layout(
                    title=f"{model_type} Clustering with User-Defined Parameters",
                    xaxis_title="Dimension 1",
                    yaxis_title="Dimension 2",
                    height=600,
                    showlegend=False
                )

                # Display the interactive cluster plot
                st.plotly_chart(fig)



        # Feature Importance: Visualize with a heatmap or bar plot
        if st.checkbox("Feature Importance"):
            # Only use numeric columns for correlation analysis
            numeric_columns = df.select_dtypes(include=[np.number])

            if numeric_columns.shape[1] > 1:
                st.markdown("### Feature Importance Heatmap")
                corr = numeric_columns.corr()
                plt.figure(figsize=(10, 8))
                sns.heatmap(corr, annot=True, cmap='coolwarm')
                st.pyplot(plt)
            else:
                st.write("No sufficient numeric columns for feature importance analysis.")


        # Clustering Interface: Allow input for new data and cluster it, then recommend similar articles
        if st.checkbox("Clustering Interface"):
            # User enters a title or keywords
            new_data = st.text_area("Enter a title or keywords for recommendations:")

            if new_data:
                # Step 1: Convert the input text into numerical features using TF-IDF
                vectorizer = TfidfVectorizer(max_features=1000)  # Adjust max_features as needed
                all_titles = df['title'].tolist()  # Use titles from the input data (df)
                all_titles.append(new_data)  # Add the new input to the list for vectorization

                tfidf_matrix = vectorizer.fit_transform(all_titles)
                new_data_vector = tfidf_matrix[-1]  # The vectorized form of the new input
                existing_data_vectors = tfidf_matrix[:-1]  # All other vectorized titles

                # Step 2: Apply PCA to reduce the number of features to match the model's expectation (e.g., 2)
                pca = PCA(n_components=2)  # Assuming the model was trained on 2 features
                pca.fit(existing_data_vectors.toarray())  # Fit PCA on the existing data vectors
                new_data_pca = pca.transform(new_data_vector.toarray())  # Reduce new input to 2 features
                existing_data_pca = pca.transform(existing_data_vectors.toarray())  # Reduce existing data to 2 features

                # Step 3: Find the cluster of the new input using the clustering model (GMM in this case)
                # Load the clustering model
                with open('gmm_model.pkl', 'rb') as f:
                    gmm_model = pickle.load(f)

                # Predict the cluster of the new input
                new_data_cluster = gmm_model.predict(new_data_pca)[0]
                st.write(f"The new input belongs to cluster: {new_data_cluster}")

                # Step 4: Recommend articles from the same cluster
                st.write(f"Articles from the same cluster ({new_data_cluster}):")
                cluster_indices = (y_labels == new_data_cluster)  # Find all articles in the same cluster
                same_cluster_df = df[cluster_indices]

                # Get the original index from the full dataframe where cluster match occurs
                cluster_original_indices = df.index[cluster_indices]

                # Step 5: Find the top 5 most similar articles based on cosine similarity
                cosine_similarities = cosine_similarity(new_data_pca, existing_data_pca).flatten()

                # Get top 5 most similar articles, but first filter for only articles in the same cluster
                top_similar_indices = cosine_similarities.argsort()[::-1]  # Sort in descending order of similarity

                # Map back to the original indices of the articles in the same cluster
                valid_indices = [idx for idx in top_similar_indices if idx in cluster_original_indices][:5]

                if valid_indices:
                    recommended_articles = df.iloc[valid_indices]
                    st.write(recommended_articles[['title', 'date']])
                else:
                    st.write("No relevant articles found in this cluster.")


        # Comparative Analysis: Compare performance of different models
        # Comparative Analysis: Compare performance of different models using X_embedded
        if st.checkbox("Comparative Analysis"):
            st.markdown("## Comparative Analysis of Models")

            # Let the user select clustering models to compare
            model_options = ['GMM', 'Agglomerative Clustering', 'Spectral Clustering', 'Birch Clustering']
            selected_models = st.multiselect("Select models to compare:", model_options)

            # Load X_embedded for visualization
            with open('X_embedded.pkl', 'rb') as f:
                X_embedded = pickle.load(f)

            # Set up dynamic subplots based on the number of selected models
            num_models = len(selected_models)
            if num_models == 0:
                st.write("Please select at least one model for comparison.")
            else:
                # Create subplots: dynamically create rows and columns based on the number of models selected
                rows = (num_models + 1) // 2  # 2 plots per row
                fig = make_subplots(rows=rows, cols=2, subplot_titles=selected_models)

                row_idx, col_idx = 1, 1  # Initialize row and column indices

                # Lists to store the metrics for each model
                silhouette_scores = []
                davies_bouldin_scores = []
                calinski_harabasz_scores = []

                for model_type in selected_models:
                    if model_type == 'GMM':
                        with open('gmm_model.pkl', 'rb') as f:
                            gmm_model = pickle.load(f)
                        predicted_labels = gmm_model.predict(X_embedded)

                    elif model_type == 'Agglomerative Clustering':
                        with open('agg_model.pkl', 'rb') as f:
                            agg_model = pickle.load(f)
                        predicted_labels = agg_model.fit_predict(X_embedded)

                    elif model_type == 'Spectral Clustering':
                        with open('spectral_model.pkl', 'rb') as f:
                            spectral_model = pickle.load(f)
                        predicted_labels = spectral_model.fit_predict(X_embedded)

                    elif model_type == 'Birch Clustering':
                        with open('birch_model.pkl', 'rb') as f:
                            birch_model = pickle.load(f)
                        predicted_labels = birch_model.predict(X_embedded)

                    # Calculate clustering metrics
                    silhouette_avg = silhouette_score(X_embedded, predicted_labels)
                    davies_bouldin = davies_bouldin_score(X_embedded, predicted_labels)
                    calinski_harabasz = calinski_harabasz_score(X_embedded, predicted_labels)

                    # Append the scores to the lists
                    silhouette_scores.append(silhouette_avg)
                    davies_bouldin_scores.append(davies_bouldin)
                    calinski_harabasz_scores.append(calinski_harabasz)

                    # Add a new scatter plot for the current model
                    scatter = go.Scatter(
                        x=X_embedded[:, 0],
                        y=X_embedded[:, 1],
                        mode='markers',
                        marker=dict(
                            size=6,
                            color=predicted_labels,
                            colorscale='Viridis',
                            showscale=False
                        ),
                        name=model_type
                    )

                    # Add scatter plot to the appropriate subplot
                    fig.add_trace(scatter, row=row_idx, col=col_idx)

                    # Update row and column indices for the next plot
                    if col_idx == 1:
                        col_idx = 2
                    else:
                        col_idx = 1
                        row_idx += 1

                # Update layout of the subplots
                fig.update_layout(
                    title_text="Comparative Cluster Analysis",
                    height=400 * rows,  # Adjust the height dynamically based on the number of rows
                    showlegend=False
                )

                # Display the interactive subplot
                st.plotly_chart(fig)

                # Only display the metrics bar charts if there are models selected
                if selected_models:
                    st.markdown("### Performance Metrics Comparison")

                    # Create Silhouette Score bar chart
                    fig_silhouette = go.Figure(
                        data=[go.Bar(x=selected_models, y=silhouette_scores, marker_color='blue')],
                        layout=go.Layout(title="Silhouette Score", xaxis_title="Models", yaxis_title="Score")
                    )
                    st.plotly_chart(fig_silhouette)

                    # Create Davies-Bouldin Index bar chart (lower is better)
                    fig_davies = go.Figure(
                        data=[go.Bar(x=selected_models, y=davies_bouldin_scores, marker_color='red')],
                        layout=go.Layout(title="Davies-Bouldin Index", xaxis_title="Models", yaxis_title="Score (lower is better)")
                    )
                    st.plotly_chart(fig_davies)

                    # Create Calinski-Harabasz Index bar chart (higher is better)
                    fig_calinski = go.Figure(
                        data=[go.Bar(x=selected_models, y=calinski_harabasz_scores, marker_color='green')],
                        layout=go.Layout(title="Calinski-Harabasz Index", xaxis_title="Models", yaxis_title="Score")
                    )
                    st.plotly_chart(fig_calinski)

    else:
        st.write("Please upload a dataset to proceed.")
