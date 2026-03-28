             Machine Learning Project Report: Clothing Size Prediction
Dataset Overview

This project is based on the ModCloth dataset, which contains customer-related information used for predicting clothing sizes. The dataset includes both numerical and categorical features, where numerical columns consist of body measurements such as height, weight, bust, waist, and hips, while categorical columns include variables like fit, category, and brand. The dataset contains multiple records (rows) and several feature columns, making it suitable for machine learning applications. This dataset was selected because it represents a real-world problem in e-commerce, where incorrect clothing size selection leads to high return rates and customer dissatisfaction. The primary purpose of using this dataset is to build an intelligent system that can recommend accurate clothing sizes based on customer body measurements, ultimately improving user experience and reducing operational costs.

 Model Objective and Approach

The main objective of this project is to develop a classification model that predicts clothing size categories such as Small (S), Medium (M), Large (L), Extra Large (XL), and Extra Extra Large (XXL). Initially, the size variable existed in numerical form, but it was transformed into categorical labels using quantile-based binning. This conversion allowed the problem to be treated as a classification task instead of regression, making it more aligned with real-world scenarios. Multiple machine learning models were trained, including CatBoost, LightGBM, Random Forest, Support Vector Machine (SVM), and XGBoost. Among these models, CatBoost was selected as the final model because it achieved the best performance in terms of accuracy and overall balance across evaluation metrics. The model works by learning patterns between body measurements and clothing sizes and then predicting the most appropriate size category for new inputs.

                            Working of Dataset 

 Data Cleaning Process

The working process of this project began with data cleaning to ensure data quality and consistency. In this phase, unnecessary columns such as item_id, review_summary, and review_text were removed because they were not relevant to the prediction task. Missing values in numerical features were handled using median imputation, as it is robust to skewed data, while missing categorical values were filled using the mode. Duplicate records were identified and removed to avoid bias in model training, and the dataset index was reset for proper alignment. Outliers were detected using the Interquartile Range (IQR) method, which identifies extreme values based on statistical dispersion. Instead of removing outliers, capping (clipping) was applied to limit extreme values within acceptable bounds. This approach preserved important information while reducing noise, resulting in a clean, consistent, and reliable dataset.

 Exploratory Data Analysis (EDA)

After data cleaning, Exploratory Data Analysis (EDA) was performed to understand the dataset in depth. Statistical measures such as mean, variance, skewness, and kurtosis were calculated to analyze the distribution of features. Histograms and Kernel Density Estimation (KDE) plots were used to visualize data distribution. To test normality, the Kolmogorov-Smirnov test was applied, along with Q-Q plots and comparisons between Empirical Cumulative Distribution Function (ECDF) and Normal CDF. The results indicated that most features were not normally distributed and exhibited skewness. Correlation analysis was performed using Spearman correlation to identify relationships between features and the target variable, as it is suitable for non-linear data. Heatmaps were used to visualize correlations, while boxplots helped detect variability and outliers. Donut charts were used to analyze class imbalance, revealing that the dataset had moderate imbalance. These insights helped in understanding feature importance and overall data behavior.

 Feature Engineering

To improve model performance, feature engineering was applied by creating new meaningful features that better represent body proportions. These included Body Mass Index (BMI), calculated as weight divided by height squared, as well as ratios such as height-to-weight ratio, chest-to-waist ratio, and waist-to-hip ratio. These engineered features helped capture hidden patterns in the data. Categorical variables were converted into numerical form using Label Encoding, and numerical features were scaled using Standard Scaling to ensure consistent feature ranges. This step enhanced the dataset’s quality and made it more suitable for model training.

 Feature Selection

Feature selection was performed to reduce redundancy and improve efficiency. First, correlation filtering was applied to remove highly correlated features with values greater than 0.9, which helps eliminate duplicate information. Next, the Variance Inflation Factor (VIF) method was used to detect multicollinearity, and features with VIF values greater than 5 were removed to ensure independence among variables. Finally, Random Forest feature importance was used to select the most relevant features contributing to size prediction. This process reduced dimensionality, improved model efficiency, and minimized overfitting.

 Train-Test Split

After preprocessing, the dataset was divided into training and testing sets using an 80/20 ratio. This ensured that the model was trained on the majority of the data and evaluated on unseen data. Special care was taken to avoid data leakage, ensuring that the test data remained independent from the training process. This step provided a reliable framework for evaluating model performance.

 Model Training and Comparison

In the model training phase, multiple classification algorithms were trained and compared. CatBoost was highly effective due to its ability to handle categorical data efficiently and reduce overfitting. LightGBM provided fast and efficient training, Random Forest offered robustness and interpretability, SVM performed well in high-dimensional spaces, and XGBoost delivered strong gradient boosting performance. These models were evaluated using accuracy, precision, recall, and F1 score. After cross-validation, CatBoost achieved the best results with high accuracy and balanced performance across all metrics. The final CatBoost model was trained using 300 iterations, a depth of 6, and a learning rate of 0.1, resulting in stable and reliable predictions.

 Model Evaluation and Interpretation

Further evaluation of the model included advanced techniques such as the confusion matrix, which showed how accurately each class was predicted. The ROC curve was used to measure the model’s ability to distinguish between classes, while the Precision-Recall curve provided insights into performance on imbalanced data. Additionally, SHAP (SHapley Additive exPlanations) analysis was used to interpret the model’s predictions by identifying the contribution of each feature. This made the model more transparent and explainable.

Final Outcome and Conclusion

In conclusion, this project successfully developed a machine learning-based clothing size prediction system that can accurately classify customer sizes based on body measurements. The system effectively handles real-world noisy data, identifies important features, and provides reliable and interpretable predictions. The use of advanced preprocessing techniques, statistical analysis, feature engineering, and model comparison ensured strong performance. This solution can be practically applied in e-commerce platforms to improve size recommendations, reduce product returns, and enhance customer satisfaction.