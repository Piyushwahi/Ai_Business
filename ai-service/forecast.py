import yfinance as yf
import pandas as pd
from prophet import Prophet
import io, base64
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def map_idea_to_ticker(idea):
    mapping = {
        "ecommerce": "XLY",   # Consumer Discretionary ETF
        "ai": "QQQ",          # Nasdaq Tech ETF
        "finance": "XLF",     
        "health": "XLV",
        "energy": "XLE"
    }
    for key, ticker in mapping.items():
        if key in idea.lower():
            return ticker
    return "SPY"

def forecast_sector(idea):
    ticker_symbol = map_idea_to_ticker(idea)

    raw = yf.download(ticker_symbol, period="5y", interval="1d")["Close"].dropna()

    # FIX: ensure Series, not DataFrame
    if isinstance(raw, pd.DataFrame):
        raw = raw.iloc[:, 0]
    data = raw.squeeze()

    if data.empty:
        raise ValueError(f"No data found for ticker {ticker_symbol}")

    df = pd.DataFrame({"ds": data.index, "y": data.values})
    model = Prophet(yearly_seasonality=True, daily_seasonality=False)
    model.fit(df)

    future = model.make_future_dataframe(periods=365)
    forecast = model.predict(future)

    last_price = df["y"].iloc[-1]
    future_price = forecast["yhat"].iloc[-1]
    change_pct = ((future_price - last_price) / last_price) * 100

    trend = "Increase" if change_pct > 0 else "Decrease"
    confidence = round(abs(change_pct), 2)

    # Plot
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(df["ds"], df["y"], label="Historical", color="blue")
    ax.plot(forecast["ds"], forecast["yhat"], label="Forecast", color="orange")
    ax.fill_between(forecast["ds"], forecast["yhat_lower"], forecast["yhat_upper"], alpha=0.2)
    ax.set_title(f"{ticker_symbol} Forecast")
    ax.legend()

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    chart_base64 = base64.b64encode(buf.read()).decode()

    return chart_base64, trend, confidence
