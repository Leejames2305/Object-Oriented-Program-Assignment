'''
Object-Oriented Programming Assignment
Student 1: Lee XY
Student 2: Chua YJ

Title: Transaction analysis with Pandas and visualisation using Matplotlib
'''
import pandas as pd
import matplotlib.pyplot as plt


# Part 1: Loading dataset, data cleaning on transactions record
with open('transactions.csv', 'r') as file:
    transactions = pd.read_csv(file)

    # Print summary (Total transactions before cleaning)
    print(f"Total transactions before data cleaning: {len(transactions)}")

    # Remove rows with errors (as long as errors contain any value)
    transactions = transactions[~transactions['errors'].notna()]
    # Remove rows with id, mcc or amount missing
    transactions = transactions.dropna(subset=['id', 'mcc', 'amount'])
    # Remove rows with amount equal to 0
    transactions['amount'] = transactions['amount'].replace('[\\$,]', '', regex=True).astype(float)
    transactions = transactions[transactions['amount'] != 0]
    # Remove duplicates based on id, mcc and amount (Retain the last occurrence)
    transactions = transactions.drop_duplicates(subset=['id', 'mcc', 'amount'], keep='last')
    
    # Print summary (Total transactions after cleaning)
    print(f"Total transactions after data cleaning: {len(transactions)}")
    
    # DEBUG: Print the cleaned transactions dataset 
    # print(transactions)


# Part 2: Data Merging and management
with open('merchant_codes.csv', 'r') as file:
    merchant_codes = pd.read_csv(file)

with open('fraud.csv', 'r') as file:
    fraud = pd.read_csv(file)

# Based on 'mcc', merge transactions with Business_Type (left join)
transactions = transactions.merge(merchant_codes[['mcc', 'Business_Type']], on='mcc', how='left', indicator=False)

# Based on 'id', merge transactions with Fraud_Status (left join)
transactions = transactions.merge(fraud[['id', 'Fraud_Status']], on='id', how='left', indicator=False)

# Fill missing values in 'merchant_name' and 'fraud_label' with 'Unknown'
transactions['Business_Type'] = transactions['Business_Type'].fillna('Unknown')
transactions['Fraud_Status'] = transactions['Fraud_Status'].fillna('Unknown')

# Based on amount, represent credit with credit_debit column (Example: $14.57 = credit, $-77.00 = debit)
# transactions['amount'] = transactions['amount'].replace('[\\$,]', '', regex=True).astype(float)
transactions['credit_debit'] = transactions['amount'].apply(lambda x: 'credit' if x >= 0 else 'debit')

# Print summary (Transactions summary after merging)
print(f"Total transactions: {len(transactions)}")
print(f"Number of debit transactions: {len(transactions[transactions['credit_debit'] == 'debit'])}")
print(f"Number of credit transactions: {len(transactions[transactions['credit_debit'] == 'credit'])}")
print(f"Fraud status breakdown:")
print(f"  - Number of fraudulent transactions: {len(transactions[transactions['Fraud_Status'] == 'Yes'])}")
print(f"  - Number of legitimate transactions: {len(transactions[transactions['Fraud_Status'] == 'No'])}")
print(f"  - Number of unknown transactions: {len(transactions[transactions['Fraud_Status'] == 'Unknown'])}")
print(f"Business type summary:")
print(f"  - Number of business types: {transactions['Business_Type'].nunique()}")
print(f"  - Number of business types involve in fraudulent transactions: {transactions[transactions['Fraud_Status'] == 'Yes']['Business_Type'].nunique()}")
print(f"  - Number of business types involve in legitimate transactions: {transactions[transactions['Fraud_Status'] == 'No']['Business_Type'].nunique()}")
print(f"  - Number of business types involve in unknown transactions: {transactions[transactions['Fraud_Status'] == 'Unknown']['Business_Type'].nunique()}")

# DEBUG: Print the merged transactions dataset
# print(transactions)


# Part 3: Data analysis and Visualisation
# Average Transaction Amount ($) per Date ; Include all transactions fraud legit and unknown
transactions['date'] = pd.to_datetime(transactions['date'])
avg_amount_per_date = transactions.groupby(transactions['date'].dt.date)['amount'].mean()

# Fraud Rate (%) by Hour of Day
# Fraud Rate = (Number of Fraudulent Transactions at that hour/ Total Transactions at that hour) * 100
transactions['hour'] = transactions['date'].dt.hour
fraud_rate_by_hour = transactions.groupby('hour')['Fraud_Status'].apply(lambda x: (x == 'Yes').sum() / len(x) * 100)

# Overall Fraud Status Distribution ; Display percentage and transactions of each fraud status category
fraud_status_distribution = transactions['Fraud_Status'].value_counts(normalize=True) * 100
fraud_status_count = transactions['Fraud_Status'].value_counts()

# Debit vs Credit Transaction by Fraud Status ; Display percentage and transactions count of each payment type for each fraud status
debit_credit_by_fraud_distribution = transactions.groupby('Fraud_Status')['credit_debit'].value_counts(normalize=True) * 100
debit_credit_by_fraud_count = transactions.groupby(['Fraud_Status', 'credit_debit']).size().unstack(fill_value=0)

# Top 5 Business types by Fraudulent Transactions Amount ; Display top 5 business types, and show its total fraudulent transactions $
top_business_types_by_fraud_amount = transactions[transactions['Fraud_Status'] == 'Yes'].groupby('Business_Type')['amount'].sum().nlargest(5)

# Visualisation - 5 Subplots  (TODO: Fix incorrect subplot layout and wrong data display)
fig, axs = plt.subplots(2, 3, figsize=(18, 10))
# Subplot 1: Average Transaction Amount ($) per Date
axs[0, 0].bar(avg_amount_per_date.index, avg_amount_per_date.values, color='blue')
axs[0, 0].set_title('Average Transaction Amount ($) per Date')
axs[0, 0].set_xlabel('Date')
axs[0, 0].set_ylabel('Average Amount ($)')
axs[0, 0].tick_params(axis='x', rotation=45)

# Subplot 2: Fraud Rate (%) by Hour of Day
axs[0, 1].bar(fraud_rate_by_hour.index, fraud_rate_by_hour.values, color='orange')
axs[0, 1].set_title('Fraud Rate (%) by Hour of Day')
axs[0, 1].set_xlabel('Hour of Day')
axs[0, 1].set_ylabel('Fraud Rate (%)')

# Subplot 3: Overall Fraud Status Distribution
axs[0, 2].pie(fraud_status_distribution.values, labels=fraud_status_distribution.index, autopct='%1.1f%%', startangle=140)
axs[0, 2].set_title('Overall Fraud Status Distribution')

# Subplot 4: Debit vs Credit Transaction by Fraud Status
axs[1, 0].bar(debit_credit_by_fraud_count.index, debit_credit_by_fraud_count['debit'], label='Debit', color='red')
axs[1, 0].bar(debit_credit_by_fraud_count.index, debit_credit_by_fraud_count['credit'], bottom=debit_credit_by_fraud_count['debit'], label='Credit', color='green')
axs[1, 0].set_title('Debit vs Credit Transaction by Fraud Status')
axs[1, 0].set_xlabel('Fraud Status')
axs[1, 0].set_ylabel('Number of Transactions')
axs[1, 0].legend()

# Subplot 5: Top 5 Business types by Fraudulent Transactions Amount
top_business_types_by_fraud_amount.plot(kind='bar', ax=axs[1, 1], color='purple')
axs[1, 1].set_title('Top 5 Business Types by Fraudulent Transactions Amount')
axs[1, 1].set_xlabel('Business Type')
axs[1, 1].set_ylabel('Total Fraudulent Amount ($)')

# Adjust layout and show the plots
plt.tight_layout()
plt.show()
