from oauth2client.service_account import ServiceAccountCredentials
import gspread
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error

pd.set_option('display.max_rows', None)  # No limit on the number of rows displayed
pd.set_option('display.max_columns', None)  # No limit on the number of columns
pd.set_option('display.width', None)  # Automatically adjust the display width
pd.set_option('display.max_colwidth', None)  # Display full content of each column


def run_arima_model(data, category, test_size=0.2):
    # Filter data for the selected category
    category_data = data[data['Categories'] == category]

    # Aggregate data by Date and resample to daily frequency
    category_data = category_data.groupby('Date')['Price'].sum().resample('D').sum().fillna(0)

    # Split the data into training and test sets
    split_point = int(len(category_data) * (1 - test_size))
    train, test = category_data[:split_point], category_data[split_point:]

    # Fit the model on the training set
    model = ARIMA(train, order=(1,1,1))
    model_fit = model.fit()

    # Forecast the same number of steps as in the test set
    predictions = model_fit.forecast(steps=len(test))

    # Calculate accuracy metrics
    mae = mean_absolute_error(test, predictions)
    rmse = np.sqrt(mean_squared_error(test, predictions))

    return predictions, mae, rmse



def remove_suffix(date):
    return date.replace('st', '').replace('nd', '').replace('rd', '').replace('th', '')


scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name("PATH", scope)

# Authenticate with Google Sheets
client = gspread.authorize(creds)

# Open Google Sheet by title
sheet = client.open_by_key("KEY")
sheet_list = sheet.worksheets()
sheet_names = [sheet.title for sheet in sheet_list]
selected_sheet_name = st.sidebar.selectbox('Select a Sheet', sheet_names)

# Select the worksheet by title 
worksheet = sheet.worksheet(selected_sheet_name)

# Get all values from the worksheet as a list of lists
data = worksheet.get_all_values()

df = pd.DataFrame(data)
df = df.replace('', np.nan)

# Drop rows with all NaN values
df.dropna(how='all', inplace=True)

# Drop columns with all NaN values
df.dropna(axis=1, how='all', inplace=True)

total_incomes_row = df[df[0] == 'Total Additonal Incomes']
total_incomes_value = total_incomes_row.iloc[0, 1]
print(total_incomes_value)

metro_spending_total = df[df[0] == 'Metro Total Spending']
metro_spending_total = total_incomes_row.iloc[0, 1]
print(metro_spending_total)


for i, row in enumerate(data):
    if "Description" in row and "Categories" in row and "Price" in row and "Date" in row:
        start_row_index = i + 1  # Start from the next row
    elif not any(row):  # Empty row indicates the end of spending data
        end_row_index = i
        break

# Filter out the relevant spending data
spending_data = data[start_row_index:end_row_index]

new_column_names = ['New_Description', 'New_Categories', 'New_Price', 'New_Date']

# Create a DataFrame from the spending data
spending_df = pd.DataFrame(spending_data[1:])
spending_df = spending_df.loc[:, (spending_df != '').any(axis=0)]


new_column_names = ['Description', 'Categories', 'Price', 'Date']

# Assign the new column names to the DataFrame
spending_df.columns = new_column_names

spending_df['Description'] = spending_df['Description'].astype(str)
spending_df['Categories'] = spending_df['Categories'].astype(str)

# Remove "$" sign and convert "Price" column to integers
spending_df['Price'] = spending_df['Price'].str.replace('$', '').astype(float).astype(int)

# Convert "Date" column to datetime objects
spending_df['Date'] = pd.to_datetime(spending_df['Date'].apply(remove_suffix) + ' 2023', format='%b %d %Y')



st.title('Spending Dashboard')

# Dropdown for category selection
category = st.sidebar.selectbox('Select Category', spending_df['Categories'].unique())

# Filter data based on selection
filtered_data = spending_df[spending_df['Categories'] == category]

# Display data table
st.write('Data Table')
st.write(filtered_data)

# Plotting with Plotly
st.write(f'Spending Over Time - {category}')
fig = px.bar(filtered_data, x='Date', y='Price', title=f'Spending in {category} Over Time')
st.plotly_chart(fig)

# Predictive Analysis
if st.button('Predict Next Period Spending and Show Accuracy'):
    try:
        prediction, mae, rmse = run_arima_model(spending_df, category)
        next_day_prediction = prediction.iloc[-1]
        st.write(f'Predicted Spending for next week in {category}: ${next_day_prediction:.2f}')
        st.write(f'Model Accuracy: MAE = {mae:.2f}, RMSE = {rmse:.2f}')
    except Exception as e:
        st.error(f"Error in prediction: {e}")