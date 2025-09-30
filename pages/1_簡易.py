# Updated content of pages/1_簡易.py

# Helper function to format the latest summary date

def format_latest_date(latest_summary):
    if latest_summary and 'timestamp' in latest_summary:
        return latest_summary['timestamp'].strftime('%Y-%m-%d')
    return '最新'

# Existing code logic...
# Assuming latest_summary is already loaded in the code

# Replace static headings with dynamic ones
latest_summary = ...  # Load your latest_summary here

# Generate dynamic headings
latest_date_label = format_latest_date(latest_summary)

# Replace the headings as follows:
# f'### 最終精選（{latest_date_label}）'
# f'### 買進報告（{latest_date_label}，前2檔）'

# Keep all other logic and layout unchanged