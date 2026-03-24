             Machine Learning Project Report: Clothing Size Prediction
 Dataset Overview

The dataset used in this project is the ModCloth dataset, which contains detailed customer-related information including body measurements, product attributes, and user reviews. It includes both numerical features such as height, weight, bust, and waist measurements, as well as categorical features like product category, fit type, and brand. This dataset was selected because it represents a real-world e-commerce scenario where customers frequently face difficulty in selecting the correct clothing size. Incorrect size selection often leads to high return rates and customer dissatisfaction. Therefore, this dataset is highly relevant for building an intelligent system that can assist in predicting accurate clothing sizes. The outcome of using this dataset is that it provides meaningful patterns between body measurements and clothing sizes, which can be learned by machine learning models to make reliable predictions.

 Project Objective
SIZE PREDICTION

The primary objective of this project is to develop a machine learning model capable of predicting clothing size categories such as Small (S), Medium (M), Large (L), Extra Large (XL), and Extra Extra Large (XXL). The model uses customer body measurements and related features as input to determine the most appropriate size category. This transformation of the problem into a classification task allows the system to provide clear and usable outputs for real-world applications, particularly in online shopping platforms.

 WORKING PROCESS

🔹 1. DATA CLEANING

The first step in the project involved data cleaning to ensure the dataset was accurate and suitable for analysis. Unnecessary columns such as item_id, review_summary, and review_text were removed because they do not contribute to size prediction and may introduce noise into the model. Missing values were handled carefully by filling numerical columns with their median values and categorical columns with their mode values, ensuring that no information was lost while maintaining data consistency. Duplicate records were removed to avoid bias in the model, and the dataset index was reset to maintain proper structure. Outliers were detected using the Interquartile Range (IQR) method, and instead of removing them completely, they were capped within acceptable limits to preserve data while reducing extreme variations. These steps were necessary because raw data often contains inconsistencies that can negatively affect model performance. As a result, the dataset became clean, consistent, and free from missing values, leading to improved reliability and reduced noise in further analysis.

 2. EXPLORATORY DATA ANALYSIS (EDA)

After cleaning, exploratory data analysis was performed to understand the dataset in depth. Various statistical measures such as mean, variance, skewness, and kurtosis were calculated to analyze the distribution of numerical features. Visualizations such as histograms and KDE plots were used to examine the shape of data distributions. Normality tests, including the Kolmogorov-Smirnov test, Q-Q plots, and comparisons between empirical and theoretical distributions, were conducted to assess whether the data followed a normal distribution. Additionally, Spearman correlation analysis was used to evaluate the relationship between features and the target variable. Heatmaps, boxplots, and donut charts were created to visualize correlations and class distributions. This step was essential to identify patterns, detect skewness, and understand relationships between variables. The results showed that most features were not normally distributed, some features had strong correlations with the target variable, and the dataset exhibited moderate class imbalance.

 3. FEATURE ENGINEERING

Feature engineering was applied to enhance the predictive power of the dataset by creating new meaningful variables. New features such as Body Mass Index (BMI), height-to-weight ratio, chest-to-waist ratio, and waist-to-hip ratio were derived from existing measurements. These features provide deeper insights into body proportions, which are directly related to clothing size. In addition, categorical variables were converted into numerical format using label encoding, and numerical features were standardized using scaling techniques. These steps were necessary to ensure that the model could effectively interpret and learn from the data. As a result, the dataset became more informative, allowing the model to capture hidden relationships and improve prediction accuracy.

 4. FEATURE SELECTION

Feature selection was performed to reduce redundancy and improve model efficiency. Initially, highly correlated features with a correlation greater than 0.9 were removed to eliminate duplicate information. Then, Variance Inflation Factor (VIF) analysis was conducted to detect and remove multicollinearity among features, ensuring that no feature overly influenced another. Finally, a Random Forest model was used to calculate feature importance, and only the most relevant features were selected using a median threshold. These steps were crucial to prevent overfitting, reduce computational complexity, and improve model performance. The result was a refined and optimized set of features that contributed significantly to accurate predictions.

 5. TARGET TRANSFORMATION

The original size variable was numerical, which was not ideal for classification tasks. Therefore, it was transformed into categorical size classes (S, M, L, XL, XXL) using quantile-based binning. This transformation converted the problem from regression into classification, making it easier for machine learning models to predict discrete size categories. The result was a well-structured target variable suitable for classification algorithms.

 6. TRAIN-TEST SPLIT

The dataset was divided into training and testing sets, with 80% of the data used for training and 20% for testing. This split ensures that the model is evaluated on unseen data, providing a realistic measure of its performance. A data leakage check was also performed to ensure that no overlapping data existed between training and testing sets. The result confirmed that there was no data leakage, ensuring a valid and reliable evaluation process.

TOP 5 CLASSIFIER COMPARISON

In order to identify the best-performing model, five different machine learning classifiers were implemented and compared: CatBoost, LightGBM, Random Forest, Support Vector Machine (SVM), and XGBoost. These models were selected because they represent a mix of advanced ensemble methods and traditional algorithms. CatBoost was chosen for its ability to handle categorical data effectively, LightGBM for its speed and efficiency, Random Forest for its robustness and interpretability, SVM for handling high-dimensional data, and XGBoost for its powerful gradient boosting capabilities.

Each model was evaluated using multiple performance metrics, including accuracy, precision, recall, and F1 score, through cross-validation. This ensured that the comparison was fair and reliable. The results showed that all models performed reasonably well, but CatBoost consistently achieved the highest accuracy along with balanced precision, recall, and F1 scores.

 BEST MODEL: CATBOOST

Based on the comparison results, CatBoost was selected as the best-performing model. This model was chosen because it handles categorical features efficiently, reduces overfitting, and delivers strong performance on complex datasets. The model was trained using 300 iterations, a depth of 6, and a learning rate of 0.1, which provided an optimal balance between learning and generalization.

The evaluation results demonstrated high accuracy along with balanced precision, recall, and F1 scores, indicating that the model performs well across all classes. Additional analysis was conducted to further validate the model. The confusion matrix provided insight into classification performance for each class, while the ROC curve and precision-recall curve illustrated the model's ability to distinguish between classes. Furthermore, SHAP analysis was used to interpret the model by showing how each feature contributes to predictions, making the model more transparent and explainable.

 FINAL OUTCOME

The final outcome of this project is a fully developed machine learning system capable of accurately predicting clothing size categories based on customer data. The system successfully handles real-world challenges such as missing values, outliers, and imbalanced data. It identifies the most important features influencing size prediction and provides interpretable results using advanced techniques like SHAP.

Overall, this project demonstrates a complete and well-structured machine learning pipeline, starting from raw data processing to final model deployment. The use of CatBoost as the final model ensures high performance and reliability, making this solution suitable for real-world applications such as online clothing size recommendation systems.