# Integrating-MLOps-for-Predictive-and-Recommender-Systems-in-Travel-Project
Productionization of ML Systems

# Integrating-MLOps-for-Predictive-and-Recommender-Systems-in-Travel

Business Context

In the realm of travel and tourism, the intersection of data analytics and machine learning presents an opportunity to revolutionize the way travel experiences are curated and delivered. This capstone project revolves around a trio of datasets - users, flights, and hotels - each providing a unique perspective on travel patterns and preferences. The goal is to leverage these datasets to build and deploy sophisticated machine learning models, serving a dual purpose: enhancing predictive capabilities in travel-related decision-making and mastering the art of MLOps through hands-on application.

Brief overview of each datasets:

Users Dataset:

code: User identifier.

company: Associated company.

name: Name of the user.

gender: Gender of the user.

age: Age of the user.

Flights Dataset:

travelCode: Identifier for the travel.

userCode: User identifier(linked to the Users dataset)

from: Origin of the flight.

to: Destination of the flight.

flightType: Type of flight (e.g., first class).

price: Price of the flight.

time: Flight duration.

distance: Distance of the flight.

agency: Flight agency.

Hotels Dataset:

travelCode: Identifier for the travel, similar to the Flights dataset.

userCode: User identifier(linked to the Users dataset)

name: Name of the hotel.

place: Location of the hotel.

days: Number of days of the hotel stay.

price: Price per day.

total: Total price for the stay.

date: Date of the hotel booking.

Project Objectives:

1. Regression Model Development:

Build a regression model to predict the price of a flight using the flights.csv dataset. Focus on feature selection, model training, and validation to ensure accuracy and reliability.

2. REST API for Regression Model:

Develop a REST API using Flask to serve the flight price prediction model, enabling real-time price predictions.

3. Containerization:

Package and deploy the flight price prediction model using Docker, ensuring portability and ease of deployment.

4. Kubernetes for Scalability:

Deploy the model using Kubernetes to manage scalability and handle varying loads efficiently.

5. Automated Workflows with Apache Airflow:

Design and implement automated workflows for managing the travel data, specifically for the regression models. Develop Directed Acyclic Graphs (DAGs) to orchestrate complex workflows in an efficient and manageable way.

6. CI/CD Pipeline with Jenkins:

Implement a Continuous Integration/Continuous Deployment (CI/CD) pipeline using Jenkins for consistent and reliable deployment of the travel price prediction model.

7. Model Tracking with MLFlow:

Utilize MLFlow for tracking and managing different versions of the travel price prediction model, ensuring a systematic approach to model iteration and deployment.

8. Gender Classification Model:

Deploy a classification model to categorize a user's gender.

9. Travel Recommendation Model:

Build a recommendation model to provide hotel suggestions based on user preferences and historical data. Develop a Streamlit web application to display insights and visualizations derived from the deployed travel recommendation model, offering an interactive and user-friendly interface for data exploration.




