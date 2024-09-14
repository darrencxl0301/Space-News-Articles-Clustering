# Space-News-Articles-Clustering

Overview
This application is designed to demonstrate clustering and visualization of space news articles using various algorithms such as Gaussian Mixture Model (GMM), Agglomerative Clustering, Spectral Clustering, and Birch Clustering. It features an interactive web-based dashboard built with Streamlit that allows users to explore and visualize the clustering results, perform data explorations, and compare different clustering models.

## Features
Home Page: Displays clusters of space news articles and allows filtering by cluster and keywords.

Dashboard: Interactive tools for data uploading, PCA visualization, dynamic clustering, and comparative analysis of different clustering models.

Interactive Charts: Users can dynamically interact with data visualizations.

Clustering Interface: Allows input of new data for clustering and provides recommendations for similar articles.
Installation

## Clone the repository:
git clone https://your-repository-url.git
cd your-project-directory
Set up a virtual environment (Optional but recommended):


python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
Install the required packages:


pip install -r requirements.txt

## Running the Application
### Start the Streamlit application:

streamlit run app.py
This command will start the Streamlit server and open your default web browser to the URL where the app is hosted, typically http://localhost:8501.

Navigate the application using the sidebar to explore different functionalities and visualizations.

## Usage
Use the sidebar to switch between the "Home" and "Dashboard" pages.

On the "Home" page, interact with the clustering results by adjusting the cluster number and entering keywords.

On the "Dashboard" page, upload your dataset, explore data, visualize PCA results, and compare the performance of various clustering models.

Dynamic charts and model performance metrics can be explored for in-depth analysis.


## License
GPL-3.0 license

### Feel free to customize this README to better fit your project's specific context or to add additional sections that you might find necessary, such as 'Contact Information', 'Acknowledgments', or 'Further Reading'.
