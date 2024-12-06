# Predictive Analytics for Detecting Fraudulent Transactions in Credit Card Data

## 1. Title and Author

- **Project Title:** Predictive Analytics for Detecting Fraudulent Transactions in Credit Card Data
- **Prepared for:** UMBC Data Science Master Degree Capstone by Dr. Chaojie (Jay) Wang
- **Author:** Sirisha Gathpa
- **GitHub:** https://github.com/sirishagathpa48
- **LinkedIn:** https://www.linkedin.com/in/sirishagathpa48/
- **PowerPoint Presentation:** [PPT Link](https://github.com/sirishagathpa48/UMBC-DATA606-Capstone/blob/main/docs/Predictive%20Analytics%20for%20Detecting%20Fraudulent%20Transactions%20in%20Credit%20Card%20Data.pptx)
- **Streamlit App:** https://creditfraudprediction.streamlit.app/
- **YouTube Video:** [Link]

## 2. Background
- **What is it about?**
- The dataset focuses on **fraud detection in financial transactions**. It provides information on various attributes related to each transaction, such as customer details, transaction amounts, merchant locations, and whether the transaction was classified as fraudulent. Fraud detection in financial systems is vital for minimizing risks and protecting financial institutions and customers from economic losses due to fraudulent activities. 

- In the field of **financial fraud detection**, machine learning has been increasingly adopted due to its ability to process large volumes of data efficiently and detect suspicious patterns. Fraudulent activities in the financial sector can include unauthorized purchases, identity theft, and card cloning, all of which can cause significant damage to both customers and financial organizations.
    
- **why is it matters?**
- Financial fraud is a growing global issue, with **billions of dollars lost every year** due to fraudulent transactions. Fraud detection systems that can automatically and accurately identify suspicious activities help financial institutions:
- **Minimize financial losses** from unauthorized transactions.
- **Protect customers** from the consequences of fraud, such as identity theft or the use of their personal information.
- **Enhance trust** in the banking system and online transactions, leading to better customer relationships and satisfaction.
- **Meet regulatory requirements** for financial security and fraud prevention.

Traditional rule-based systems are often not effective enough to catch evolving fraud patterns. **Machine learning models** offer a more flexible and adaptive approach to identify fraud by learning from historical transaction data and recognizing new, unseen patterns. These models can evolve with the nature of fraud, reducing false positives while catching true fraud cases.
     
- **What are your research questions?**
- 1. Can machine learning models accurately classify fraudulent transactions using the features available in this dataset?
- 2. Which features contribute the most to the accurate prediction of fraud?
- 3. How do the geographic locations of cardholders and merchants affect the likelihood of fraud?
- 4. What patterns emerge from analyzing fraud across categories, jobs, and states?

- #### Relevance in Industry
- Fraud detection is an important **real-time problem** in the banking and retail industries, where millions of transactions happen daily. Large-scale payment processors, credit card companies, and e-commerce platforms require efficient fraud detection systems that can prevent fraudulent transactions before they are completed. Machine learning models can quickly process incoming transactions, flagging potential fraud within seconds, thus minimizing damage.

## 3. Data

#### Data sources:
https://www.kaggle.com/code/akhilpm1996/credit-card-fraud-prediction/input

#### Data size:
- **143.64 MB**

#### Data shape:
- **555,719 rows** and **23 columns**

#### Time period:
- **June 2020** (based on `trans_date_trans_time`)

#### What does each row represent?
Each row represents a **financial transaction** between a cardholder and a merchant.

#### Data dictionary:
| Column Name           | Data Type     | Definition                                              | Potential Values                            |
|-----------------------|---------------|---------------------------------------------------------|---------------------------------------------|
| `trans_date_trans_time`| DateTime      | The date and time of the transaction                    | Date and time format                        |
| `cc_num`              | Numeric       | Credit card number (masked)                             | Numeric (Scientific Notation)               |
| `merchant`            | String        | Merchant's name where the transaction took place        | Text                                        |
| `category`            | String        | Category of the transaction                             | `personal_care`, etc.                       |
| `amt`                 | Float         | Transaction amount                                      | Numeric                                     |
| `first`               | String        | Customer's first name                                   | Text                                        |
| `last`                | String        | Customer's last name                                    | Text                                        |
| `gender`              | String        | Gender of the customer                                  | `M`, `F`                                    |
| `street`              | String        | Customer's street address                               | Text                                        |
| `city`                | String        | City where the customer resides                         | Text                                        |
| `state`               | String        | State where the customer resides                        | `SC`, etc.                                  |
| `zip`                 | Integer       | Zip code of the customer’s address                      | 5-digit number                              |
| `lat`                 | Float         | Latitude of the customer's location                     | Numeric                                     |
| `long`                | Float         | Longitude of the customer's location                    | Numeric                                     |
| `city_pop`            | Integer       | Population of the customer’s city                       | Numeric                                     |
| `job`                 | String        | Job of the customer                                     | Text (`Mechanical engineer`, etc.)          |
| `dob`                 | DateTime      | Date of birth of the customer                           | Date format                                 |
| `trans_num`           | String        | Unique transaction number                               | Alphanumeric                                |
| `unix_time`           | Float         | Transaction timestamp in UNIX format                    | Numeric (Scientific Notation)               |
| `merch_lat`           | Float         | Latitude of the merchant’s location                     | Numeric                                     |
| `merch_long`          | Float         | Longitude of the merchant’s location                    | Numeric                                     |
| `is_fraud`            | Integer       | Whether the transaction was fraudulent                  | `0` (not fraud), `1` (fraud)                |

#### Potential values (for categorical variables):
- **Category**: personal_care, health_fitness, travel, etc.

- **Is_fraud**: 1 (fraud), 0 (legitimate).

#### Target/Label:
- **is_fraud** will be the target variable in the machine learning model.

#### Features/Predictors:
- Potential features for the ML model include:
  - **amt** (Transaction amount)
  - **category** (Transaction type)
  - **state** (City population)
  - **trans_date_trans_time** (The date and time of the transaction)
  - **lat** (Latitude of the customer's location )
  - **long** (Longitude of the customer's location)
  - **job** (Job title)

## 4. Project Outcome
- **Fraud Prediction:** Machine learning models can be trained on this dataset to predict whether a transaction is fraudulent.
- **Fraud Probability Estimation:** The models estimate the likelihood of a transaction being fraudulent based on features like transaction category, amount, time, and user demographics.

## 4. Exploratory Data Analysis
- **Cleaned the dataset by droping the null values and replacing some of the column null values with the mean values**
- **Splitted the 'trans_date_trans_time' column into 'trans_date' and 'trans_time' to extract and create a new column for transaction date,time,month and dayofweek**
- **Dropped 'street', 'zip', 'city_pop', 'trans_num', 'unix_time', 'merch_lat', 'merch_long','first','last','dob' columns as they are not relevant**
- **Used Plotly.express for visualization**

### Visualizations
- **Visualized data to explore distributions and compare fraudulent vs. non-fraudulent transactions, leading to feature selection for the model.**
#### 4.1. Distribution of target variable.
- Distribution of transactions shows that most of them are Non-fraudulent (99.6%) while Fraudulent(0.38%) among the total data.
- It shows imbalanced data.

![image](https://github.com/sirishagathpa48/UMBC-DATA606-Capstone/blob/main/docs/Images/FraudvsNon-FraudTransactions_Image1.png)

#### 4.2. Distribution of transactions by months
- For 'Not Fraud' cases, June 2020 had the lowest number of transactions, followed by a steady increase, peaking in December. The significant rise in December aligns with the holiday season, particularly Christmas, and the typical year-end boost in consumer demand.

- For fraudulent transactions, July had the fewest cases, with a steady increase reaching a peak in August. After August, fraud transactions gradually declined, though remained relatively high from August to October.

![image](https://github.com/sirishagathpa48/UMBC-DATA606-Capstone/blob/main/docs/Images/TransactionByMonth_Image2.png)

#### 4.3. Distribution of transactions by day of week
- The two charts show a similar pattern, with Sunday, Monday, and Tuesday having the highest number of transactions for 
both fraud and non-fraud cases.This indicates we should pay more attentions to transactions happen on these at as 
they are more likely to be fraud.

![image](https://github.com/sirishagathpa48/UMBC-DATA606-Capstone/blob/main/docs/Images/TransactionByDayOfWeek_Image3.png)

#### 4.4. Distribution of transactions by Gender
- In both fraudulent and non-fraudulent transactions, females conduct more transactions than males, although the difference is not very large.

![image](https://github.com/sirishagathpa48/UMBC-DATA606-Capstone/blob/main/docs/Images/TransactionByGender_Image4.png)

#### 4.5. Distribution of transactions by Category
- For non-fraudulent transactions, the top three categories are gas_transport, grocery_pos, and home, with gas_transport being the highest. For fraudulent transactions, the leading categories are grocery_pos, shopping_net, and misc_net. Notably, grocery_pos appears in both categories, indicating it warrants closer scrutiny.

![image](https://github.com/sirishagathpa48/UMBC-DATA606-Capstone/blob/main/docs/Images/TransactionByCategory_Image5.png)

#### 4.6. Distribution of transactions by AgeGroup
- For non-fraudulent transactions, the top three categories are gas_transport, grocery_pos, and home, with gas_transport being the highest. For fraudulent transactions, the leading categories are grocery_pos, shopping_net, and misc_net. Notably, grocery_pos appears in both categories, indicating it warrants closer scrutiny.

![image](https://github.com/sirishagathpa48/UMBC-DATA606-Capstone/blob/main/docs/Images/TransactionByAgeGroup_Image6.png)

#### 4.7. Distribution of transactions by AgeGroup
- From 12 AM to 11 AM, the number of transactions remains relatively stable. From 11 AM to midnight, there's a noticeable increase in transaction activity, indicating that people are more active during this period. For fraudulent transactions, most occur late at night (10 PM to midnight) or early in the morning (12 AM to 4 AM), suggesting that individuals with malicious intent are more likely to act during times of reduced human monitoring.

![image](https://github.com/sirishagathpa48/UMBC-DATA606-Capstone/blob/main/docs/Images/TransactionByTime_Image7.png)
  
#### 4.8. Distribution of top 10 transactions and transaction amount by job
  
![image](https://github.com/sirishagathpa48/UMBC-DATA606-Capstone/blob/main/docs/Images/Top10TransactionsByJob_Image8.png)

![image](https://github.com/sirishagathpa48/UMBC-DATA606-Capstone/blob/main/docs/Images/Top10TransactionAmountByJob_Image9.png)

#### 4.9. Distribution of top 10 transactions and transaction amount by state

![image](https://github.com/sirishagathpa48/UMBC-DATA606-Capstone/blob/main/docs/Images/Top10TransactionsByState_Image10.png)

![image](https://github.com/sirishagathpa48/UMBC-DATA606-Capstone/blob/main/docs/Images/Top10TransactionAmountByState_Image11.png)

#### 4.10. HeatMap

![image](https://github.com/sirishagathpa48/UMBC-DATA606-Capstone/blob/main/docs/Images/HeatMap_Image12.png)

## 5. Model Building
 - Based on the Visualizations outputs dropped irrelevant columns 'id','trans_date_trans_time','cc_num', 'merchant', 'gender','city','trans_time_group', 'age', 
  'is_not_fraud','fraud_label' for the model training.
 - Considering list of job values from the above fraud and non-fraud graphs for 'Top 10 transaction count' and 'Top 10 transaction amount' where observing high number of transaction also have high number of amount
 - Standardizing  feature columns like 'Job','Category','State','DayofWeek', 'Month' and 'hour' by applying the encoder to conver the categorical data to numerical and ensured they are scaled properly for modeling purposes.
 - The dataset is divided into training and testing sets using the train_test_split method. The model is then trained on the training data using the fit method.
 - Splitting the data into 80 - 20 for training and testing.
 - Models used are Logistic regression, Decision tree, Random forest, XGboost and AdaBoost.
 - **Python packages to be used scikit-learn, Pandas,Numpy
 - **The development environment used is JUPYTER NOTEBOOK
   
### 5.1 Imbalanced Data
- Class imbalance refers to a situation where one class (e.g., fraudulent transactions) is significantly underrepresented compared to the other class (e.g., non-fraudulent transactions) in the dataset. Addressing this imbalance is crucial in machine learning, especially when the minority class is of particular importance.
- To handle the class imbalance, we applied SMOTE (Synthetic Minority Over-sampling Technique).
- SMOTE generates synthetic samples for the minority class (fraudulent transactions) to balance the dataset, ensuring the model is trained effectively to detect fraud without bias toward the majority class.
  
#### 5.2. Models with SMOTE
- Logistic Regression has a significantly lower accuracy (92.25%) compared to other models, which may suggest limitations in handling the complexity of the dataset even after SMOTE sampling.
- Random Forest achieves the highest accuracy (99.63%), demonstrating its robust performance in detecting fraudulent transactions.
- Decision Tree and XGBoost also show high accuracy levels (95.41% and 99.32%, respectively), making them effective options for fraud detection.
- AdaBoost performs well with an accuracy of 96.17%, although it is slightly outperformed by Random Forest and XGBoost.
- These results highlight Random Forest as the most effective model for this dataset, with XGBoost closely following.

  ![image](https://github.com/sirishagathpa48/UMBC-DATA606-Capstone/blob/main/docs/Images/ModelAccuracy_Image13.jpg)

## 6. Application of the Trained Models using Streamlit Web App
- This Streamlit web app predicts the likelihood of fraudulent credit card transactions using a pre-trained Random Forest model. Users input transaction details through an interactive interface, which are then processed and fed into the model. The app displays whether the transaction is potentially fraudulent or not, offering a simple and effective tool for fraud detection.

   ![image](https://github.com/sirishagathpa48/UMBC-DATA606-Capstone/blob/main/docs/Images/ModelAccuracy_Image13.jpg)

 - User Experience: Here is how the interface design helps users detect fraud and visualize patterns effectively.

   ![image](https://github.com/user-attachments/assets/e9431c1d-81cb-42e5-a16d-94469c705ad0)

## 7. Future Use of Machine Learning in Credit Card Fraud Prediction
- **Enhanced Data Integration:** Leveraging diverse sources like user behavior and device details for better fraud detection.
- **Advanced Algorithms:** Using deep learning and hybrid models to improve accuracy and detect complex patterns.
- **Real-Time Systems:** Implementing dynamic risk scoring and instant transaction analysis for proactive fraud prevention.
  
## 8. Limitations
- Despite the robust approach and high performance demonstrated by the implemented models, there are several limitations to consider:

- The models rely heavily on the quality and completeness of the dataset; any missing or erroneous data can significantly impact the accuracy of fraud detection.
- The use of SMOTE for addressing class imbalances, while effective, may introduce synthetic patterns that do not fully represent real-world fraudulent transactions.
- The model's performance depends heavily on the provided features. Any changes in transaction data structure or the introduction of new features would require re-training and re-validation of the models.
- Some models, such as Random Forest, though highly accurate, can make interpreting the results more complex due to their ensemble nature, potentially reducing transparency in decision-making.

## 9. Conclusion
- Our analysis revealed that the Random Forest classifier outperformed other models, demonstrating the highest accuracy in detecting fraudulent credit card transactions, even after addressing class imbalance with SMOTE sampling.
- Random Forest, with its ensemble approach, excelled in both accuracy and reliability, making it an effective choice for fraud detection.
- Its ability to handle complex structured data and reduce overfitting resulted in superior performance on both training and testing datasets.
- A credit card fraud detection system can significantly reduce financial losses for banks and payment processors, thereby improving financial security and customer trust.
- Logistic Regression, while effective for binary classification, was less capable of capturing the intricate patterns in fraud detection, providing simpler interpretations.
- XGBoost, though highly effective, performed slightly below Random Forest in this project but remains a strong choice for datasets requiring advanced gradient boosting techniques.
