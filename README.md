# Return Distribution & EGARCH Dashboard

A Streamlit app to upload OHLC price data (CSV), compute return distributions, and fit an EGARCH model.

## Expected CSV Columns
- `Date` (string or datetime)
- `Close` (float)
- Optional: `USD Rate` (float), `Open`, `High`, `Low`, `Volume`

## How to Run
```bash
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Embed into Canva
1. Deploy this app to a cloud host (e.g., Streamlit Community Cloud, Render, Railway, or your server).
2. In Canva, add an **Embed** and paste your app's URL, or build a simple Canva App/iframe to the hosted URL.
3. The app supports file upload directly in the sidebar.

## Notes
- Returns are computed as log or simple; EGARCH is fit on returns *scaled to percent* so the conditional volatility and forecasts are in percent.
- You can toggle currency conversion using the optional `USD Rate` column.