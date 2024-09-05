# Online Payment Fraud Detection

- Author - Sirisha Gathpa
- Semester - Fall'24
- Prepared for UMBC Data Science Master Degree Capstone by Dr Chaojie (Jay) Wang
- GitHub - https://github.com/sirishagathpa48
- LinkedIn - https://www.linkedin.com/in/sirishagathpa48/
- 
---

### Background

#### What is it about?
This dataset is about financial transactions and their potential classification as either fraudulent or legitimate. By analyzing the features associated with each transaction, machine learning models can be trained to detect fraudulent behavior in real-time.

#### Why does it matter?
Fraud detection is crucial in reducing financial losses for institutions and protecting consumers from identity theft and unauthorized transactions. By leveraging machine learning, fraud detection systems can improve accuracy and efficiency, helping businesses minimize risks.

#### What are your research questions?
1. Can machine learning models accurately classify fraudulent transactions using the features available in this dataset?
2. Which features contribute the most to the accurate prediction of fraud?
3. How do the geographic locations of cardholders and merchants affect the likelihood of fraud?

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
1. **trans_date_trans_time**: `object` - Timestamp of the transaction (date and time).
2. **cc_num**: `float64` - Unique customer identification number.
3. **merchant**: `object` - The merchant involved in the transaction.
4. **category**: `object` - Transaction type (e.g., personal care, travel).
5. **amt**: `float64` - Transaction amount.
6. **first**: `object` - Cardholder's first name.
7. **last**: `object` - Cardholder's last name.
8. **gender**: `object` - Cardholder's gender.
9. **street**: `object` - Cardholder's street address.
10. **city**: `object` - Cardholder's city of residence.
11. **state**: `object` - Cardholder's state of residence.
12. **zip**: `int64` - Cardholder's zip code.
13. **lat**: `float64` - Latitude of cardholder's location.
14. **long**: `float64` - Longitude of cardholder's location.
15. **city_pop**: `int64` - Population of the cardholder's city.
16. **job**: `object` - Cardholder's job title.
17. **dob**: `object` - Cardholder's date of birth.
18. **trans_num**: `object` - Unique transaction identifier.
19. **unix_time**: `int64` - Transaction timestamp in Unix format.
20. **merch_lat**: `float64` - Merchant's location (latitude).
21. **merch_long**: `float64` - Merchant's location (longitude).
22. **is_fraud**: `int64` - Fraudulent transaction indicator (1 = fraud, 0 = legitimate).

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


