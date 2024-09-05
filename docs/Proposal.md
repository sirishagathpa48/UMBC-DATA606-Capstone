# Online Payment Fraud Detection

- Author - Sirisha Gathpa
- Semester - Fall'24
- Prepared for UMBC Data Science Master Degree Capstone by Dr Chaojie (Jay) Wang
- GitHub - https://github.com/sirishagathpa48
- LinkedIn - https://www.linkedin.com/in/sirishagathpa48/
  
---

### Background

#### What is it about?
The dataset focuses on **fraud detection in financial transactions**. It provides information on various attributes related to each transaction, such as customer details, transaction amounts, merchant locations, and whether the transaction was classified as fraudulent. Fraud detection in financial systems is vital for minimizing risks and protecting financial institutions and customers from economic losses due to fraudulent activities. 

In the field of **financial fraud detection**, machine learning has been increasingly adopted due to its ability to process large volumes of data efficiently and detect suspicious patterns. Fraudulent activities in the financial sector can include unauthorized purchases, identity theft, and card cloning, all of which can cause significant damage to both customers and financial organizations.

#### Why does it matter?
Financial fraud is a growing global issue, with **billions of dollars lost every year** due to fraudulent transactions. Fraud detection systems that can automatically and accurately identify suspicious activities help financial institutions:
- **Minimize financial losses** from unauthorized transactions.
- **Protect customers** from the consequences of fraud, such as identity theft or the use of their personal information.
- **Enhance trust** in the banking system and online transactions, leading to better customer relationships and satisfaction.
- **Meet regulatory requirements** for financial security and fraud prevention.

Traditional rule-based systems are often not effective enough to catch evolving fraud patterns. **Machine learning models** offer a more flexible and adaptive approach to identify fraud by learning from historical transaction data and recognizing new, unseen patterns. These models can evolve with the nature of fraud, reducing false positives while catching true fraud cases.

#### Research Questions
1. Can machine learning models accurately classify fraudulent transactions using the features available in this dataset?
2. Which features contribute the most to the accurate prediction of fraud?
3. How do the geographic locations of cardholders and merchants affect the likelihood of fraud?
4. What are the challenges in balancing detection accuracy and false positives?

#### Relevance in Industry
Fraud detection is an important **real-time problem** in the banking and retail industries, where millions of transactions happen daily. Large-scale payment processors, credit card companies, and e-commerce platforms require efficient fraud detection systems that can prevent fraudulent transactions before they are completed. Machine learning models can quickly process incoming transactions, flagging potential fraud within seconds, thus minimizing damage.

---

### Data

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
- **Gender**: M (Male), F (Female).
- **Is_fraud**: 1 (fraud), 0 (legitimate).

#### Target/Label:
- **is_fraud** will be the target variable in the machine learning model.

#### Features/Predictors:
- Potential features for the ML model include:
  - **amt** (Transaction amount)
  - **category** (Transaction type)
  - **city_pop** (City population)
  - **cc_num** (Customer ID number)
  - **gender** (Gender)
  - **merch_lat** (Merchant location latitude)
  - **merch_long** (Merchant location longitude)
  - **job** (Job title)


