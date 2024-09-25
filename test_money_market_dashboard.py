import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
import requests
import json
from datetime import datetime, timedelta

# Layout configuration
st.set_page_config(layout="wide", page_title="Money Markets")

# Title of the App
st.title("Money Markets Dashboard")

# Define the FredPy class (assuming this has been implemented in the background)

start = '1900-01-01'  
end = datetime.now().strftime('%Y-%m-%d')
# Initialize FRED with your API key
API_KEY = 'a2fb338b4ef6e2dcb7c667c21b2d1c4e'

end_date = datetime.now()
start_date = end_date - timedelta(days=365*10)


# Define the FredPy class
class FredPy:

    def __init__(self, token=None):
        self.token = token
        self.url = "https://api.stlouisfed.org/fred/series/observations" + \
                    "?series_id={seriesID}&api_key={key}&file_type=json" + \
                    "&observation_start={start}&observation_end={end}&units={units}"

    def set_token(self, token):
        self.token = token

    def get_series(self, seriesID, start, end, units='lin'):
        # Format the URL with the values
        url_formatted = self.url.format(
            seriesID=seriesID, start=start, end=end, units=units, key=self.token
        )
        
        # Request the data from FRED API
        response = requests.get(url_formatted)
        
        if self.token:
            if response.status_code == 200:
                # Extract and format the data as a DataFrame
                data = pd.DataFrame(response.json()['observations'])[['date', 'value']] \
                        .assign(date=lambda cols: pd.to_datetime(cols['date'])) \
                        .assign(value=lambda cols: pd.to_numeric(cols['value'], errors='coerce')) \
                        .rename(columns={'value': seriesID})
                
                # This will convert non-numeric values (e.g., '.') to NaN
                return data
            else:
                print(f"Error: Bad response from API, status code = {response.status_code}")
                print(f"URL: {url_formatted}")
                print(f"Response content: {response.content}")
                raise Exception(f"Bad response from API, status code = {response.status_code}")
        else:
            raise Exception("You did not specify an API key.")

# Instantiate FredPy object
fredpy = FredPy()
fredpy.set_token(API_KEY)

def get_indicators(df, start, end):
    
    # Initialize an empty DataFrame to store all indicators
    macro_indicators = pd.DataFrame()

    # Loop through each indicator, fetch the data, and merge it into the main DataFrame
    for name, series_id in df.items():
        print(f"Fetching data for {name} ({series_id})")
        
        # Fetch the series data using the get_series method
        try:
            series_data = fredpy.get_series(
                seriesID=series_id,
                start=start,
                end=end
            )
            
            # Rename the 'value' column to the name of the series (key)
            series_data = series_data.rename(columns={series_id: name})
            
            # Merge the series data with the macro_indicators DataFrame
            if macro_indicators.empty:
                macro_indicators = series_data.set_index('date')
            else:
                macro_indicators = macro_indicators.merge(series_data.set_index('date'), on='date', how='outer')

        except Exception as e:
            print(f"Failed to fetch data for {name} ({series_id}): {e}")

    # Display the first few rows of the final DataFrame
    return macro_indicators


def generate_sector_board(sectors_dict, start, end):
    """
    This function generates a sector performance board for given sectors over specific time periods.

    Parameters:
    - sectors_dict: Dictionary where keys are sector tickers and values are sector names.
    - start: Start date for downloading the data.
    - end: End date for downloading the data.

    Returns:
    - A styled DataFrame with sector performance metrics and percentile ranks.
    """
    # Download adjusted close prices for the given tickers in sectors_dict
    sectors_data = yf.download(tickers=list(sectors_dict.keys()), start=start, end=end)['Adj Close']

    # Initialize an empty DataFrame for the board
    sector_board = pd.DataFrame()

    # Add the 'Current Price' column
    sector_board['Current Price'] = sectors_data.iloc[-1]

    # Calculate performance metrics
    current_performance = sectors_data.pct_change(periods=252).iloc[-1]
    mean_1y_perf = sectors_data.pct_change(periods=252).mean()  # Mean of the 1-year performance
    std_1y_perf = sectors_data.pct_change(periods=252).std()    # Standard deviation of the 1-year performance
    z_score = (current_performance - mean_1y_perf) / std_1y_perf

    # Add performance columns for different periods (1 week, 1 month, 3 months, 1 year)
    sector_board = pd.concat([sector_board,
                              sectors_data.pct_change(periods=6).iloc[-1].rename('1 week Perf'),
                              sectors_data.pct_change(periods=21).iloc[-1].rename('1 month Perf'),
                              sectors_data.pct_change(periods=63).iloc[-1].rename('3 month Perf'),
                              sectors_data.pct_change(periods=252).iloc[-1].rename('1 year Perf'),
                              z_score.rename('Z-Score')],
                             axis=1)

    # Create the 'Asset' column by mapping sector names from sectors_dict
    sector_board['Asset'] = sector_board.index.map(sectors_dict)

    # Reorder columns to have 'Asset' as the first column
    sector_board = sector_board[['Asset', 'Current Price', '1 week Perf', '1 month Perf', '3 month Perf', '1 year Perf',
                                'Z-Score']]

    # Format the DataFrame for better readability
    sector_board = sector_board.style.format({
        'Current Price': "{:.2f}",
        '1 week Perf': "{:.2%}",
        '1 month Perf': "{:.2%}",
        '3 month Perf': "{:.2%}",
        '1 year Perf': "{:.2%}",
        'Z-Score': "{:.2f}"
    })

    return sector_board

def generate_macro_board(macro_dict, start, end):
    """
    This function generates a macro performance board for given macro indicators over specific time periods.

    Parameters:
    - macro_dict: Dictionary where keys are macro indicator names and values are their FRED API codes.
    - start: Start date for downloading the data.
    - end: End date for downloading the data.

    Returns:
    - A styled DataFrame with macro performance metrics and percentile rank for 1-year performance.
    """
    # Download the macro indicator data
    macro_data = get_indicators(macro_dict, start=start, end=end)

    # Ensure the index is DatetimeIndex before resampling
    if not isinstance(macro_data.index, pd.DatetimeIndex):
        macro_data.index = pd.to_datetime(macro_data.index)

    # Resample monthly and forward-fill missing values
    macro_data = macro_data.resample('M').last().ffill()

    # Initialize an empty DataFrame for the board
    macro_board = pd.DataFrame()

    # Add the 'Current Value' column
    macro_board['Current Value'] = macro_data.iloc[-1]

    # Calculate performance metrics for different periods
    current_performance = macro_data.pct_change(periods=12).iloc[-1]  # 1-year performance
    mean_1y_perf = macro_data.pct_change(periods=12).mean()  # Mean of the 1-year performance
    std_1y_perf = macro_data.pct_change(periods=12).std()    # Standard deviation of the 1-year performance
    z_score = (current_performance - mean_1y_perf) / std_1y_perf

    # Add performance columns for different periods (1 month, 3 months, 1 year)
    macro_board = pd.concat([macro_board,
                             macro_data.pct_change(periods=1).iloc[-1].rename('1 month Perf'),
                             macro_data.pct_change(periods=3).iloc[-1].rename('3 month Perf'),
                             macro_data.pct_change(periods=12).iloc[-1].rename('1 year Perf'),
                             z_score.rename('Z-Score')],
                            axis=1)
    
    # 3-month rate of change (3m ROC of 1 year performance)
    macro_board['3m 1y Perf ROC'] = macro_board['1 year Perf'] - macro_board['1 year Perf'].shift(3)
    
    # Handle NaN values caused by shifting (e.g., the first few rows might be NaN)
    macro_board['3m 1y Perf ROC'] = macro_board['3m 1y Perf ROC'].fillna(0)  # Or use another method

    # Calculate the percentile rank for 1-year performance
    macro_board['1 year Perf Percentile Rank'] = macro_board['1 year Perf'].rank(pct=True)

    # Reorder columns
    macro_board = macro_board[['Current Value', '1 month Perf', '3 month Perf', '1 year Perf', 'Z-Score']]

    # Format the DataFrame for better readability
    macro_board = macro_board.style.format({
        'Current Value': "{:.2f}",
        '1 month Perf': "{:.2%}",
        '3 month Perf': "{:.2%}",
        '1 year Perf': "{:.2%}",
        'Z-Score': "{:.2f}"
    })

    # Return the styled DataFrame
    return macro_board


bonds = {
    '10y-2y Spread': 'T10Y2Y',
    '10y-3mo Spread': 'T10Y3M',
    '10y Treasury Yield': 'DGS10',
    'Fed Funds Rate': 'FEDFUNDS',
    '1 month T-yield': 'WGS1MO',
    '3 month T-yield': 'WGS3MO',
    '6 month T-yield': 'WGS6MO',
    '1 year T-yield': 'WGS1YR',
    '2 year T-yield': 'WGS2YR',
    '3 year T-yield': 'WGS3YR',
    '5 year T-yield': 'WGS5YR',
    '7 year T-yield': 'WGS7YR',
    '10 year T-yield': 'WGS10YR',
    '20 year T-yield': 'WGS20YR',
    '30 year T-yield': 'WGS30YR',
    '5 year TIPS': 'WFII5',
    '7 year TIPS': 'WFII7',
    '10 year TIPS': 'WFII10',
    '20 year TIPS': 'WFII20',
    '30 year TIPS': 'WFII30',
    'AAA Corp Yield': 'BAMLC0A1CAAAEY',
    'BBB Corp Yield': 'BAMLC0A4CBBBEY',
    'CCC Corp Yield': 'BAMLH0A3HYCEY',
    }

inflation = {
    'CPI All Items': 'CPIAUCSL',
    'CPI Ex Food & Energy':'CPILFESL',
    'PPI All Items' : 'PPIACO',
    'PPI Ex Food & Energy': 'WPSFD4131',
    'PCE Deflator' : 'PCEPI',
    'Retail Money Market Funds': 'WRMFNS'
}

fixed_income_dict = {
    'TLT': '20+ Year Treasury Bond ETF',
    'MBB': 'Mortgage-Backed Securities ETF',
    'IEF': '7-10 Year Treasury Bond ETF',
    'LQD': 'Investment Grade Corporate Bond ETF',
    'HYG': 'High Yield Corporate Bond ETF',
    'SHY': '1-3 Year Treasury Bond ETF',
    }

money_markets = get_indicators(bonds, start=start, end=end).resample('W').last().ffill()

# Adding custom bond spreads and inflation expectations
money_markets['AAA-BBB Corp Yield'] = (money_markets['AAA Corp Yield'] - money_markets['BBB Corp Yield']) * -1
money_markets['AAA-BBB Corp Yield'] =  (money_markets['AAA Corp Yield'] - money_markets['BBB Corp Yield']) *-1
money_markets['AAA-CCC Corp Yield'] =  (money_markets['AAA Corp Yield'] - money_markets['CCC Corp Yield']) *-1
money_markets['BBB-CCC Corp Yield'] =  (money_markets['BBB Corp Yield'] - money_markets['CCC Corp Yield']) *-1
money_markets['10yr-AAA Corp Yield'] =  (money_markets['10 year T-yield'] - money_markets['AAA Corp Yield']) *-1
money_markets['10yr-BBB Corp Yield'] =  (money_markets['10 year T-yield'] - money_markets['BBB Corp Yield']) *-1
money_markets['10yr-CCC Corp Yield'] =  (money_markets['10 year T-yield'] - money_markets['CCC Corp Yield']) *-1
money_markets['5yr Implied Inflation'] =  money_markets['5 year T-yield'] - money_markets['5 year TIPS']


# Define the layout with 4 columns
col1, col2 = st.columns([1, 1])
col3, col4 = st.columns([1, 1])

t_yields = money_markets[['Fed Funds Rate', '3 month T-yield','2 year T-yield',
                         '5 year T-yield','10 year T-yield', '30 year T-yield', 
                         '10y-2y Spread','AAA-BBB Corp Yield','CCC Corp Yield',
                         '5yr Implied Inflation']]
df = pd.DataFrame()

df['Current Yield'] = t_yields.iloc[-1]  # This takes the last row of each column as the current yield

df['WoW bps'] = (t_yields.iloc[-1] - t_yields.iloc[-3])  # WoW change (last row vs second last row)
df['MoM bps'] = (t_yields.iloc[-1] - t_yields.iloc[-5])  # MoM change (assuming monthly data)
df['QoQ bps'] = (t_yields.iloc[-1] - t_yields.iloc[-13])  # QoQ change (assuming quarterly data)
df['YoY bps'] = (t_yields.iloc[-1] - t_yields.iloc[-53])  # YoY change (assuming weekly data)

# Calculating the 1-year mean and standard deviation for each yield
df['1Y Mean'] = t_yields.iloc[-53:].mean()  # Mean of the last 252 rows (1 year)
df['1Y StdDev'] = t_yields.iloc[-53:].std()  # Standard deviation of the last 252 rows (1 year)

# Calculating the Z-Score for 1 year
df['1Y Z-Score'] = (df['Current Yield'] - df['1Y Mean']) / df['1Y StdDev']

# Calculating the 3-year mean and standard deviation for each yield
df['3Y Mean'] = t_yields.iloc[-159:].mean()  
df['3Y StdDev'] = t_yields.iloc[-159:].std() 

df['5Y Mean'] = t_yields.iloc[-795:].mean()
df['5Y StdDev'] = t_yields.iloc[-795:].std()

df['3Y Z-Score'] = (df['Current Yield'] - df['3Y Mean']) / df['3Y StdDev']

df['5Y Z-Score'] = (df['Current Yield'] - df['5Y Mean']) / df['5Y StdDev']

# Dropping the intermediate mean and std columns if not needed
df = df.drop(columns=['1Y Mean', '1Y StdDev', '3Y Mean', '3Y StdDev', '5Y Mean', '5Y StdDev'])

df = df.style.format({
    'Current Yield' : "{:.2f}",
    'WoW bps' : "{:.2f}",
    'QoQ bps' : "{:.2f}",
    'MoM bps' : "{:.2f}",
    'YoY bps' : "{:.2f}",
    '1Y Z-Score': "{:.2f}",
    '3Y Z-Score': "{:.2f}",
    '5Y Z-Score': "{:.2f}"
})


money_market_dashboard = df

fixed_income_data = yf.download(tickers=list(fixed_income_dict.keys()), start=start, end=end)['Adj Close']


# Bottom-left: Fixed Income Dashboard
fixed_income_dashboard = generate_sector_board(fixed_income_dict, start=start_date, end=end_date)
col2.subheader("Fixed Income Dashboard")
col2.table(fixed_income_dashboard)
inflation_dashboard = generate_macro_board(inflation, start=start, end=end)
col2.subheader("Inflation Dashboard")
col2.table(inflation_dashboard)

#########################################
# Top-right: Interactive Line Chart with Dropdown Menu
col2.subheader("Interactive Fixed Income Line Chart")

# Dropdown to select a column for fixed income data
selected_column_fixed_income = col2.selectbox("Select a fixed income column to plot", fixed_income_data.columns)

# Dropdown menu for time range selection for fixed income
time_range_fixed_income = col2.selectbox("Select a time range for fixed income", ('1Y', '5Y', '20Y', 'MAX'), index=1)  # Default is 5Y

# Filter the data based on the selected time range for fixed income
if time_range_fixed_income == '1Y':
    sliced_data_fixed_income = fixed_income_data.iloc[-252:]  # Last 52 weeks for 1 year
elif time_range_fixed_income == '5Y':
    sliced_data_fixed_income = fixed_income_data.iloc[-1260:]  # Last 260 weeks for 5 years
elif time_range_fixed_income == '20Y':
    sliced_data_fixed_income = fixed_income_data.iloc[-5040:]  # Last 1040 weeks for 20 years
else:  # 'MAX'
    sliced_data_fixed_income = fixed_income_data  # No slicing, show all data

# Calculate the 1-year moving average for the selected column for fixed income
moving_average_fixed_income = sliced_data_fixed_income[selected_column_fixed_income].rolling(window=252).mean()

# Create the Plotly figure for fixed income
line_fig_fixed_income = go.Figure()

# Add the selected fixed income column's line chart
line_fig_fixed_income.add_trace(go.Scatter(x=sliced_data_fixed_income.index, y=sliced_data_fixed_income[selected_column_fixed_income], mode='lines', name=selected_column_fixed_income))

# Add the 1-year moving average line chart for fixed income
line_fig_fixed_income.add_trace(go.Scatter(x=sliced_data_fixed_income.index, y=moving_average_fixed_income, mode='lines', name=f"{selected_column_fixed_income} 1-year MA"))

# Update the layout of the chart for fixed income
line_fig_fixed_income.update_layout(template="plotly_dark", title=f"{selected_column_fixed_income}", xaxis_title="Date", yaxis_title=selected_column_fixed_income)

# Display the plot for fixed income
col2.plotly_chart(line_fig_fixed_income)


#########################################
# Top-left: Interactive Line Chart with Dropdown Menu
col1.subheader("Interactive Money Market Line Chart")

# Dropdown to select a column for money markets
selected_column_money_market = col1.selectbox("Select a money market column to plot", money_markets.columns)

# Dropdown menu for time range selection for money markets
time_range_money_market = col1.selectbox("Select a time range for money markets", ('1Y', '5Y', '20Y', 'MAX'), index=1)  # Default is 5Y

# Filter the data based on the selected time range for money markets
if time_range_money_market == '1Y':
    sliced_data_money_market = money_markets.iloc[-52:]  # Last 52 weeks for 1 year
elif time_range_money_market == '5Y':
    sliced_data_money_market = money_markets.iloc[-260:]  # Last 260 weeks for 5 years
elif time_range_money_market == '20Y':
    sliced_data_money_market = money_markets.iloc[-1040:]  # Last 1040 weeks for 20 years
else:  # 'MAX'
    sliced_data_money_market = money_markets  # No slicing, show all data

# Calculate the 1-year moving average for the selected column for money markets
moving_average_money_market = sliced_data_money_market[selected_column_money_market].rolling(window=52).mean()

# Create the Plotly figure for money markets
line_fig_money_market = go.Figure()

# Add the selected money market column's line chart
line_fig_money_market.add_trace(go.Scatter(x=sliced_data_money_market.index, y=sliced_data_money_market[selected_column_money_market], mode='lines', name=selected_column_money_market))

# Add the 1-year moving average line chart for money markets
line_fig_money_market.add_trace(go.Scatter(x=sliced_data_money_market.index, y=moving_average_money_market, mode='lines', name=f"{selected_column_money_market} 1-year MA"))

# Update the layout of the chart for money markets
line_fig_money_market.update_layout(
    template="plotly_dark", 
    title=f"{selected_column_money_market}", 
    xaxis_title="Date", 
    yaxis_title=selected_column_money_market,
    showlegend=False  # Disable the legend
)

# Display the plot for money markets
col1.plotly_chart(line_fig_money_market)

# Top-left: Money Market Dashboard
# money_market_dashboard = money_markets
col1.subheader("Money Market Dashboard")
col1.table(money_market_dashboard)
# Bottom-right: Yield Curve Plot
yield_curve = money_markets[['Fed Funds Rate', '1 month T-yield', '3 month T-yield', '6 month T-yield', '1 year T-yield', '2 year T-yield', '3 year T-yield', '5 year T-yield', '7 year T-yield', '10 year T-yield', '20 year T-yield', '30 year T-yield']]
most_recent_curve = yield_curve.iloc[-1]
fig = go.Figure()
fig.add_trace(go.Scatter(x=yield_curve.columns, y=most_recent_curve.values, mode='lines+markers', name="Most Recent Curve"))
fig.update_layout(template="plotly_dark", title="Most Recent Yield Curve", xaxis_title="Maturity", yaxis_title="Yield (%)")
col1.subheader("Yield Curve")
col1.plotly_chart(fig)
