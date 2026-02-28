import streamlit as st

def render_knowledge():
    st.title("Machine Learning Knowledge & Model Information")
    st.markdown("Welcome to the **A²ML Knowledge Module**. This module explains ML concepts, training pipelines, models used in this system, and evaluation metrics. It serves as an interactive documentation suite.")
    
    st.markdown("---")
    
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "1. ML Overview", 
        "2. Pipeline Explanation", 
        "3. Regression", 
        "4. Classification", 
        "5. Clustering", 
        "6. Metrics", 
        "7. Feature Engineering", 
        "8. Explainable AI"
    ])
    
    with tab1:
        st.header("1. Basic Machine Learning Overview")
        st.write("### What is Machine Learning?")
        st.write("In simple terms, ML is when a system **learns from data** rather than being explicitly programmed. It identifies patterns and makes predictions on new data.")
        
        st.write("### Types of Learning")
        st.markdown("""
        - **Supervised Learning**: The algorithm learns from labeled data to predict a target variable. 
          - *Examples*: Classification (predicting discrete labels), Regression (predicting continuous values).
        - **Unsupervised Learning**: The algorithm finds patterns in unlabeled data.
          - *Examples*: Clustering (grouping similar items), Anomaly detection.
        - **Reinforcement Learning**: Learning through rewards (not used in A²ML).
        """)

    with tab2:
        st.header("2. ML Training Pipeline Explanation")
        st.write("The autonomous system replicates the process a Data Scientist follows. Here is the step-by-step pipeline:")
        st.markdown('''
        1. **Data Collection**: Loading the dataset source (e.g., uploading the CSV).
        2. **Data Preprocessing**: Handling missing values, normalizing numerical entries, and encoding categorical variables.
        3. **Feature Engineering**: Selecting the most useful features and reducing dimensionality.
        4. **Model Training**: The core process where the selected algorithms learn patterns from the processed data.
        5. **Model Evaluation**: Checking the accuracy or error rates against an unseen test set.
        6. **Prediction**: Creating the final model ready to infer answers on new data.
        ''')
        
    with tab3:
        st.header("3. Regression Models")
        st.write("Regression predicts continuous numerical values. Imagine fitting a line to data.")
        st.markdown("""
        - **Linear/Ridge/Lasso Regression**: Fits lines using different regularization penalties to prevent overfitting (L1/L2 penalties).
        - **Support Vector Regression (SVR)**: A boundary-based technique that tries to fit errors within a certain threshold boundary.
        - **Decision/Random Forest Regressor**: Uses branching rules. Random forests average out many trees.
        - **XGBoost & LightGBM (Advanced)**: Highly optimized gradient boosting frameworks that build trees sequentially to correct the errors of previous trees. Extremely powerful on tabular data.
        - **Neural Networks (MLP)**: Multi-layer Perceptrons. They attempt to model complex non-linear relationships using dense layers of artificial neurons.
        """)

    with tab4:
        st.header("4. Classification Models")
        st.write("Classification predicts distinct categories or classes.")
        st.markdown("""
        - **Support Vector Machine (SVM)**: Finds the optimal decision boundary (hyperplane) that maximally separates classes.
        - **K-Nearest Neighbor (KNN)**: Makes a prediction based on the 'k' most similar (closest) training samples.
        - **Random Forest**: Combines multiple decision trees (an ensemble) to improve accuracy and control over-fitting.
        - **XGBoost & LightGBM (Advanced)**: State-of-the-art gradient boosted trees for extreme predictive accuracy.
        - **Neural Networks (MLP Classifier)**: Uses backpropagation through hidden layers to distinguish highly complex boundaries.
        """)
        
    with tab5:
        st.header("5. Clustering Models")
        st.write("Clustering groups similar data points together without pre-labeled targets.")
        st.markdown("""
        - **K-Means**: Partitions data into 'K' distinct, non-overlapping groups by minimizing the variance within each cluster.
        - **DBSCAN**: A density-based clustering algorithm. It groups together points that are closely packed and marks low-density points as outliers.
        """)
        
    with tab6:
        st.header("6. Model Evaluation Metrics")
        st.write("How we measure if a model is 'good' or not.")
        
        st.subheader("Classification Metrics")
        st.markdown("""
        - **Accuracy**: Total correct predictions / Total predictions.
        - **Precision**: How many of the predicted positives were actually positive (minimizes false positives).
        - **Recall**: How many of the actual positives were captured (minimizes false negatives).
        - **F1 Score**: The harmonic mean of Precision and Recall. Essential for imbalanced datasets.
        - **ROC Curve (AUC)**: Area Under the Receiver Operating Characteristic curve measures predictability at different thresholds.
        """)
        
        st.subheader("Regression Metrics")
        st.markdown("""
        - **Mean Squared Error (MSE)**: The average of the squared errors between predicted and actual.
        - **Root Mean Squared Error (RMSE)**: The square root of MSE. Easier to interpret as it is in the same units as the target.
        - **R² Score**: Represents the proportion of variance in the dependent variable that is predictable.
        """)

        st.write("**Cross Validation**: Testing the model's reliability by splitting the data into multiple folds and testing repeatedly.")

    with tab7:
        st.header("7. Feature Engineering")
        st.write("Transforming raw data into features that better represent the underlying problem.")
        st.markdown("""
        - **Feature Selection**: Picking only the columns that have a strong relationship (mutual information, correlation) with the target.
        - **Dimensionality Reduction**: Shrinking the number of columns while retaining the most important variance.
        - **PCA (Principal Component Analysis)**: A mathematical procedure that transforms correlated features into a smaller number of uncorrelated variables.
        """)
        
    with tab8:
        st.header("8. Explainable AI (XAI)")
        st.write("AI models shouldn't be black boxes. We need to know *why* a model predicted a certain result.")
        st.markdown("""
        - **Feature Importance**: Showing which columns fundamentally heavily influenced the final model predictions.
        - **SHAP (SHapley Additive exPlanations)**: A game-theoretic approach to explain the output of any machine learning model. It assigns each feature an importance value for a particular prediction.
        - **Partial Dependence Plots (PDP)**: These plots show the marginal effect one or two features have on the predicted outcome of a model. While SHAP shows overall magnitude, PDP shows *directionality*—does increasing "Age" increase or decrease the "Price" prediction, and by what shape?
        """)
