# Implement a Scoring Credit Model / API / Dashboard

## **Problem to solve**

You are Data Scientist in a financial company, named **ready to spend**, which offers consumer credits for people who have little or no loan history at all.

The company wishes to implement a **scoring credit** tool to calculate the likelihood that a customer reimburses its credit, and classifies the credit application granted or refused. It therefore wishes to develop a classification algorithm by relying on varied data sources (behavioral data, data from other financial institutions, etc.).

In addition, customer relationship managers have brought up the fact that customers are increasingly seeking transparency vis-à-vis credit grant decisions. This demand for customer transparency goes quite in keeping with the values that the company wants to embody.

Ready to spend therefore decides to develop an interactive dashboard so that customer relationship managers can both explain in the most transparent possible credit decisions, but also allow their customers to have their personal information and explore them easily.

## **Your mission**


1. Building a scoring model that will give a prediction about the probability of bankruptcy of a client automatically.
2. Build an interactive dashboard for customer relationship managers to interpret the predictions made by the model, and improve customer knowledge of customer relationship loaders.
3. Select a Kernel Kaggle to facilitate the preparation of the data needed to develop the scoring model. You will analyze this kernel and adapt to make sure it meets the needs of your mission.

### **Considerations - Dashboard specifications**

This will have to contain at least the following features:

- Allow to visualize the score and interpretation of this score for each client intelligibly for a non-expert data in Data Science.
- Allow to visualize descriptive information relating to a client (via a filter system).
- Allow comparing descriptive information about a customer to all customers or group of similar clients.

#### **Deliverables**

- The interactive dashboard that meets the above specifications and the score prediction API, divided each other on the cloud.
- A folder on a code versioning tool containing:
    - The Modeling Code (pretreatment to prediction), including experiment tracking via MLFlow and centralized stocking of models
    - The code generating the dashboard
    - The code for deploying the model as an API
    - For the dashboard and API, a folder introducing the objective of the project and the folder division, and a file with all the requirements
    - A evidently report for data drift
- A methodological note describing:
    - The model training methodology (2 pages maximum)
    - The methodology to treat unbalanced data (1 page maximum)
    - The Business Cost function, the optimization algorithm and the evaluation metric (1 page maximum)
    - A summary table of the results (1 page maximum)
    - The global and local interpretation of the model (1 page maximum)
    - Limits and possible improvements (1 page maximum)
    - The data drift analysis (1 page maximum-
  - A presentation medium for defense, detailing the work done.

#### **Final work**
You can see the **final work** in the following links
- [Frontend with Streamlit](https://credit-scoring-felipelim.streamlit.app/)
- [Endpoint API](http://13.38.11.228/docs)

## **Repository file structure**
- .github/workflows: file with github actions
- api: 
- notebooks:
- dashboard:
- data/processed:
- docs:
- models:
- tests: unit tests using pytest
- presentation: 
