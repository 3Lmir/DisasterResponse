# DisasterResponse
In this project, a comprehensive analysis of a vast dataset comprising thousands of authentic messages sourced from Figure 8 is conducted. These messages were dispatched during natural disasters through various channels, including social media and direct communication with disaster response organizations. In the initial phase, objective is to construct a robust ETL (Extract, Transform, Load) pipeline for the systematic processing of message and category data stored in CSV files. The processed data will be stored in an SQLite database, serving as the foundation for our subsequent machine learning and Natural Language Processing (NLP) pipelines.

After the clean data has been stored in the database, it will be processed using Count Vectorization and TfIdf Transformation and utilized to classify messages to 36 categories using a MultiOutput Classifier. 

To note, project is still under development and in the final phase of the project, it is planned to deploy classification model within a  Flask-based web application. This application will incorporate interactive Plotly dashboards, enabling efficient data visualization and message classification.
