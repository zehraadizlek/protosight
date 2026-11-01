ProtoSight

Predicting Website Design Prototypicality Using Machine Learning

Try the Live Application

About the Project

ProtoSight is a machine learning project that predicts how closely a website's design follows common visual patterns.

For example, most online shopping websites share familiar elements, such as product images, navigation menus, and shopping categories. ProtoSight analyzes website screenshots to estimate how typical their designs are.

The goal is to explore how machine learning can help us understand website design and user expectations.

Dataset

The project uses the Web Design Prototypicality: eCommerce dataset, published by Kathrin Figl and Aliaksei Miniukovich and available through Harvard Dataverse.

The dataset contains 1,064 screenshots of e-commerce websites, including apparel and homeware stores, alongside human evaluations of their designs.

Dataset: https://doi.org/10.7910/DVN/9FKSQI

How It Works

Image Analysis: A pretrained ResNet18 neural network extracts visual features from website screenshots.

Machine Learning: Regression models learn relationships between these features and human prototypicality ratings.

Prediction: The model estimates how closely a website follows familiar design patterns.

I experimented with Ridge Regression and category-specific models, including Support Vector Regression, to evaluate different prediction approaches.

Results

The experiments produced the following results:

Model

Mean Absolute Error (MAE)

Basic baseline model

0.4965

ResNet18 + Ridge Regression

0.4687

Cross-validated model selection

0.4716

The ResNet18 + Ridge model reduced prediction error by approximately 5.6% compared to the basic baseline.

The cross-validated approach also achieved a Spearman correlation of 0.2815.

These results suggest that visual features can help predict website prototypicality, although prediction accuracy remains limited.

Try ProtoSight

The application is available online:

https://protosight.streamlit.app/

Users can explore website design prototypicality predictions through a Streamlit interface.

Running Locally

To run the application on your own computer:

Download or clone this GitHub repository.

Open the project folder in a terminal.

Install the dependencies using pip install -r requirements.txt.

Start the application using streamlit run app.py.

Open the local URL displayed in your terminal.

These instructions assume the repository contains app.py, requirements.txt, and the files needed by the prediction model.

Limitations and Future Work

Website design is subjective, and people may have different expectations depending on the website category.

ProtoSight is an experimental project rather than a definitive evaluation of website quality. Future improvements could include larger datasets, additional website categories, and more advanced image analysis models.

Data Credits

Figl, K., & Miniukovich, A. Web Design Prototypicality: eCommerce. Harvard Dataverse.

https://doi.org/10.7910/DVN/9FKSQI

Developed By

Sumeyye Zehra Dizlek
