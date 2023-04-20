import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np 

sheet_names_excel = ['2023-03-22',
 '2023-03-23',
 '2023-03-24',
 '2023-03-25',
 '2023-03-26',
 '2023-03-27',
 '2023-03-28',
 '2023-03-29',
 '2023-03-30',
 '2023-03-31']

# Read the Excel file
file_path = 'J298MCL001 - Strain Measurements First Setup.xlsx'
excel_data = pd.read_excel(file_path, sheet_name=sheet_names_excel, engine='openpyxl')
# Create a figure and axis to plot the data
# Process each sheet in the Excel file
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

all_data = np.array([])
for sheet_name, sheet_data in excel_data.items():
    # Convert the 'Date' column to datetime objects
    sheet_data['Date'] = pd.to_datetime(sheet_data['Date'])

    # Extract date and hour from the datetime objects
    sheet_data['DateOnly'] = sheet_data['Date'].dt.date
    sheet_data['Hour'] = sheet_data['Date'].dt.hour
    _a = sheet_data['DateOnly'].to_numpy()
    all_data = np.append(all_data, _a)

    # Plot the data points for the current sheet
    ax1.scatter(sheet_data['Hour'], sheet_data['DateOnly'], label=sheet_name)

ax1.set_xlabel('Hour (0-24)')
ax1.set_ylabel('Day')
ax1.yaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
ax1.xaxis.set_ticks(range(0, 25, 1))
plt.setp(ax1.get_xticklabels(), rotation=45)

unique_dates = np.unique(all_data)
sorted_unique_dates = np.sort(unique_dates)

# Create custom bins for the histogram
bins = [sorted_unique_dates[i] + pd.DateOffset(days=-0.5) for i in range(len(sorted_unique_dates))]
bins.append(sorted_unique_dates[-1] + pd.DateOffset(days=0.5))

ax2.hist(all_data, bins=bins, orientation='horizontal', align='mid', rwidth=0.8)
# ax2.hist(all_data, bins=len(sheet_names_excel), orientation='horizontal', align='mid', rwidth=0.8)
ax2.set_xlabel('Number of data points at 1Hz')

# Adjust space between subplots
plt.subplots_adjust(wspace=0.1)

plt.show()

# plt.savefig(sheet_name+'.png')
# plt.close('all')
# plt.savefig('Recorded_hours_by_date.png',dpi = 1000) 