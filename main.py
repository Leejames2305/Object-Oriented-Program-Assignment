'''
Object-Oriented Programming Assignment
Student 1: Lee XY
Student 2: Chua YJ

Title: Transaction analysis with Pandas and visualisation using Matplotlib
'''
# %%
import matplotlib  # Temp to use TkAgg for interactive
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.widgets import Button
matplotlib.use('TkAgg')  # Temp to use button in Interactive Window 

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
debit_credit_by_fraud_count = debit_credit_by_fraud_count.reindex(columns=['debit', 'credit'], fill_value=0)

# Top 5 Business types by Fraudulent Transactions Amount ; Display top 5 business types, and show its total fraudulent transactions $
top_business_types_by_fraud_amount = transactions[transactions['Fraud_Status'] == 'Yes'].groupby('Business_Type')['amount'].sum().nlargest(5)


# %%
# Part 4: Visualisation with Matplotlib
plt.style.use('seaborn-v0_8-whitegrid')
class ChartViewer:
    def __init__(
        self,
        avg_amount_per_date,
        fraud_rate_by_hour,
        fraud_status_distribution,
        fraud_status_count,
        debit_credit_by_fraud_count,
        top_business_types_by_fraud_amount,
        total_transactions,
    ):
        self.avg_amount_per_date = avg_amount_per_date
        self.fraud_rate_by_hour = fraud_rate_by_hour
        self.fraud_status_distribution = fraud_status_distribution
        self.fraud_status_count = fraud_status_count
        self.debit_credit_by_fraud_count = debit_credit_by_fraud_count
        self.top_business_types_by_fraud_amount = top_business_types_by_fraud_amount
        self.total_transactions = total_transactions
        self.label_map = {'Yes': 'Fraud', 'No': 'Legit', 'Unknown': 'Unknown'}
        self.page_titles = [
            'Average Transaction Amount ($) per Date',
            'Fraud Rate (%) by Hour of Day',
            'Overall Fraud Status Distribution',
            'Debit vs Credit Transaction by Fraud Status',
            'Top 5 Business Types by Fraudulent Transactions Amount',
        ]
        self.pages = [
            self.draw_plot_1,
            self.draw_plot_2,
            self.draw_plot_3,
            self.draw_plot_4,
            self.draw_plot_5,
        ]
        self.current_page = 0
        self.fig, self.ax = plt.subplots(figsize=(16, 9))
        self.fig.subplots_adjust(bottom=0.2, top=0.9, left=0.08, right=0.96)
        self.buttons = []
        self.create_buttons()
        self.fig.canvas.mpl_connect('key_press_event', self.handle_keypress)
        self.render()

    def create_buttons(self):
        button_layout = [
            ('Back', (0.33, 0.05, 0.1, 0.06), self.prev_page),
            ('Next', (0.45, 0.05, 0.1, 0.06), self.next_page),
            ('Exit', (0.57, 0.05, 0.1, 0.06), self.exit_viewer),
        ]
        for label, axes_position, callback in button_layout:
            button_ax = self.fig.add_axes(axes_position)
            button = Button(button_ax, label)
            button.on_clicked(callback)
            self.buttons.append(button)

    def handle_keypress(self, event):
        if event.key == 'right':
            self.next_page(event)
        elif event.key == 'left':
            self.prev_page(event)
        elif event.key in {'1', '2', '3', '4', '5'}:
            self.go_to_page(int(event.key) - 1)
        elif event.key in {'q', 'escape'}:
            self.exit_viewer(event)

    def render(self):
        self.ax.clear()
        self.ax.set_aspect('auto')
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.grid(axis='x', linestyle='--', alpha=0.35)
        self.pages[self.current_page]()
        # self.fig.suptitle(f'Chart {self.current_page + 1}/5: {self.page_titles[self.current_page]}', fontsize=14, y=0.98)
        self.fig.canvas.draw_idle()
        self.fig.subplots_adjust(bottom=0.2, top=0.9, left=0.12, right=0.96)

    def go_to_page(self, page_index):
        self.current_page = page_index
        self.render()

    def prev_page(self, _event):
        self.current_page = (self.current_page - 1) % len(self.pages)
        self.render()

    def next_page(self, _event):
        self.current_page = (self.current_page + 1) % len(self.pages)
        self.render()

    def exit_viewer(self, _event):
        plt.close(self.fig)

    def draw_plot_1(self):
        x_values = list(self.avg_amount_per_date.index)
        y_values = self.avg_amount_per_date.values.astype(float)
        self.ax.plot(x_values, y_values, color='#8aa4d6', marker='o', linewidth=2.2, alpha=0.9)
        for x_value, y_value in zip(x_values, y_values):
            self.ax.annotate(
                f'${y_value:,.2f}',
                (x_value, y_value),
                textcoords='offset points',
                xytext=(0, 10),
                ha='center',
                fontsize=8,
                fontweight='bold',
                bbox={'boxstyle': 'round,pad=0.18', 'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.82},
            )
        y_min = y_values.min()
        y_max = y_values.max()
        y_range = max(y_max - y_min, 1)
        self.ax.set_ylim(y_min - (0.35 * y_range), y_max + (0.35 * y_range))
        self.ax.set_title('Average Transaction Amount ($) per Date')
        self.ax.set_xlabel('Date')
        self.ax.set_ylabel('Average Amount ($)')
        self.ax.grid(axis='y', linestyle='--', alpha=0.35)
        self.ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        self.ax.set_xticks(x_values)
        self.ax.tick_params(axis='x', rotation=45, labelsize=6)
        self.fig.subplots_adjust(bottom=0.27)

    def draw_plot_2(self):
        x_values = list(self.fraud_rate_by_hour.index)
        y_values = self.fraud_rate_by_hour.values.astype(float)
        self.ax.bar(x_values, y_values, color='darkorange')
        max_value = max(y_values.max(), 1)
        self.ax.set_ylim(-max_value * 1.2, max_value * 1.2)
        self.ax.axhline(0, color='black', linewidth=1)
        self.ax.spines['bottom'].set_position(('data', 0))
        self.ax.spines['top'].set_visible(False)
        self.ax.set_title('Fraud Rate (%) by Hour of Day')
        self.ax.set_xlabel('Hour of Day')
        self.ax.set_ylabel('Fraud Rate (%)')
        self.ax.set_xticks(range(0, 24))
        self.ax.set_xlim(-0.5, 23.5)
        self.ax.tick_params(axis='x', rotation=0)
        self.ax.grid(axis='y', linestyle='--', alpha=0.3)
        for x_value, y_value in zip(x_values, y_values):
            self.ax.annotate(
                f'{y_value:.1f}%',
                (x_value, y_value),
                textcoords='offset points',
                xytext=(0, 6),
                ha='center',
                fontsize=8,
            )

    def draw_plot_3(self):
        labels = [self.label_map.get(l, l) for l in self.fraud_status_distribution.index]
        values = self.fraud_status_distribution.values.astype(float)
        mapped_counts = self.fraud_status_count.copy()
        mapped_counts.index = mapped_counts.index.map(self.label_map)
        total_count = int(mapped_counts.sum())
        color_map = {
            'Fraud': '#d98c8c',
            'Legit': '#95c8a0',
            'Unknown': '#b4bcc7',
        }
        colors = [color_map.get(label, '#b9c6d8') for label in labels]  # type: ignore

        explode = [0.08 if value < 1 else 0 for value in values]

        def autopct_formatter(percent):
            count = int(round((percent / 100) * total_count))
            return f'{percent:.1f}% ({count} cases)'

        wedges, text_labels, text_pcts = self.ax.pie( # type: ignore
            values,
            labels=labels,  # type: ignore
            colors=colors,
            autopct=autopct_formatter,
            startangle=140,
            pctdistance=0.5,
            labeldistance=0.55,
            explode=explode,
            wedgeprops={'linewidth': 0, 'edgecolor': 'none'},
        )
        for text in text_labels:
            text.set_fontsize(10)
            text.set_fontweight('bold')
        for text in text_pcts:
            text.set_fontsize(8)

        small_slice_index = None
        if len(values) > 0:
            smallest_value = values.min()
            if smallest_value < 1:
                small_slice_index = int(values.argmin())

        if small_slice_index is not None:
            wedge = wedges[small_slice_index]
            percent = values[small_slice_index]
            count = int(mapped_counts.loc[labels[small_slice_index]])
            angle = (wedge.theta2 + wedge.theta1) / 2.0
            x_pos = 1.04 * (np.cos(np.deg2rad(angle)))
            y_pos = 1.04 * (np.sin(np.deg2rad(angle)))
            label_x = 1.45 if x_pos >= 0 else -1.45
            label_y = y_pos * 1.15
            text_pcts[small_slice_index].set_visible(False)
            self.ax.annotate(
                f'{percent:.1f}% ({count} cases)',
                xy=(x_pos, y_pos),
                xytext=(label_x, label_y),
                textcoords='data',
                ha='left' if label_x > 0 else 'right',
                va='center',
                fontsize=9,
                fontweight='bold',
                arrowprops={'arrowstyle': '->', 'color': '#4b5563', 'lw': 1.1},
            )

        self.ax.set_title('Overall Fraud Status Distribution')
        self.ax.axis('equal')

    def draw_plot_4(self):
        counts = self.debit_credit_by_fraud_count.copy()
        counts.index = counts.index.map(self.label_map)
        preferred_order = [status for status in ['Fraud', 'Legit', 'Unknown'] if status in counts.index]
        remaining_status = [status for status in counts.index if status not in preferred_order]
        status_order = preferred_order + remaining_status
        counts = counts.reindex(status_order, fill_value=0).astype(float)
        debit_values = counts['debit'].values
        credit_values = counts['credit'].values
        y_positions = list(range(len(status_order)))

        if len(y_positions) == 0:
            self.ax.text(0.5, 0.5, 'No data available', transform=self.ax.transAxes, ha='center', va='center')
            self.ax.set_axis_off()
            return

        self.ax.barh(y_positions, debit_values, label='Debit', color="#90dfff", edgecolor='none', height=0.58)
        self.ax.barh(y_positions, credit_values, left=debit_values, label='Credit', color="#e2ff9f", edgecolor='none', height=0.58)

        max_total = max((debit_values + credit_values).max(), 1)
        self.ax.set_xlim(0, max_total * 1.58)
        offset = max_total * 0.01
        threshold = max_total * 0.05

        for idx, status in enumerate(status_order):
            debit_count = int(counts.loc[status, 'debit'])
            credit_count = int(counts.loc[status, 'credit'])

            needs_shift = (debit_count < threshold) or (credit_count < threshold)
            v_shift = 0.05 if needs_shift else 0

            if debit_count > 0:
                debit_pct = (debit_count / self.total_transactions) * 100
                self.ax.text(debit_count + offset, idx - v_shift, f'Debit: {debit_count} ({debit_pct:.1f}%)', ha='left', va='center', fontsize=8, fontweight='bold')

            if credit_count > 0:
                credit_pct = (credit_count / self.total_transactions) * 100
                self.ax.text(debit_count + credit_count + offset, idx + v_shift, f'Credit: {credit_count} ({credit_pct:.1f}%)', ha='left', va='center', fontsize=8, fontweight='bold')

        self.ax.set_yticks(y_positions)
        self.ax.set_yticklabels(status_order)
        self.ax.invert_yaxis()
        self.ax.set_title('Debit vs Credit Transaction by Fraud Status')
        self.ax.set_xlabel('Number of Transactions')
        self.ax.set_ylabel('Fraud Status')
        self.ax.legend(loc='lower right')
        self.ax.grid(axis='x', linestyle='--', alpha=0.35)

    def draw_plot_5(self):
        series = self.top_business_types_by_fraud_amount.sort_values(ascending=True)
        if series.empty:
            self.ax.text(0.5, 0.5, 'No fraudulent transactions found', transform=self.ax.transAxes, ha='center', va='center')
            self.ax.set_axis_off()
            return

        y_positions = list(range(len(series.index)))
        values = series.values.astype(float)
        self.ax.barh(y_positions, values, color='#8d9ccf', edgecolor='none', height=0.6)
        max_abs_value = max(abs(values).max(), 1)
        self.ax.set_xlim(min(0, values.min() * 1.2), max(0, values.max() * 1.2))
        self.ax.axvline(0, color='#4b5563', linewidth=1)

        for idx, amount in enumerate(values):
            offset = max_abs_value * 0.025
            label_x = amount + offset if amount >= 0 else amount - offset
            self.ax.text(
                label_x,
                idx,
                f'${amount:,.2f}',
                va='center',
                ha='left' if amount >= 0 else 'right',
                fontsize=8,
                fontweight='bold',
                color='#1f2937',
            )
        
        self.ax.set_yticks(y_positions)
        self.ax.set_yticklabels(series.index)
        self.ax.tick_params(axis='y', labelsize=8) # Replace 12 with desired size
        self.ax.invert_yaxis()
        self.ax.set_title('Top 5 Business Types by Fraudulent Transactions Amount')
        self.ax.set_xlabel('Total Fraudulent Amount ($)')
        self.ax.set_ylabel('Business Type')
        self.ax.grid(axis='x', linestyle='--', alpha=0.35)

viewer = ChartViewer(
    avg_amount_per_date=avg_amount_per_date,
    fraud_rate_by_hour=fraud_rate_by_hour,
    fraud_status_distribution=fraud_status_distribution,
    fraud_status_count=fraud_status_count,
    debit_credit_by_fraud_count=debit_credit_by_fraud_count,
    top_business_types_by_fraud_amount=top_business_types_by_fraud_amount,
    total_transactions=len(transactions),
)
plt.show()

# %%