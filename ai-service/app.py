# ai-service/app.py
import os
import io
import json
import base64
import traceback
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

import yfinance as yf
import pandas as pd
from prophet import Prophet
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import requests

load_dotenv()

AZURE_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")

app = Flask(__name__)
CORS(app)


def map_idea_to_ticker(idea: str) -> str:
    # simple mapping - extend as you like
    m = {
        "ecommerce": "AMZN",
        "ai": "QQQ",
        "finance": "XLF",
        "health": "XLV",
        "energy": "XLE",
        "garment": "XRT",
        "fashion": "XRT",
        "retail": "XRT",
    }
    it = (idea or "").lower()
    for k, v in m.items():
        if k in it:
            return v
    return "SPY"


def fetch_close_series(symbol: str, period: str = "10y") -> pd.Series:
    df = yf.download(symbol, period=period, progress=False, auto_adjust=True)
    if df is None or df.empty:
        raise ValueError(f"No data for symbol {symbol}")
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close.index = pd.to_datetime(close.index)
    return close.dropna()


def prepare_df(series: pd.Series) -> pd.DataFrame:
    df = series.reset_index()
    df.columns = ["ds", "y"]
    if pd.api.types.is_datetime64tz_dtype(df["ds"].dtype):
        df["ds"] = df["ds"].dt.tz_localize(None)
    else:
        df["ds"] = pd.to_datetime(df["ds"])
    df = df.sort_values("ds").drop_duplicates(subset=["ds"])
    df["y"] = df["y"].interpolate(method="linear").ffill().bfill()
    return df


def fit_forecast(df: pd.DataFrame, years: int = 5):
    model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    model.fit(df)
    future = model.make_future_dataframe(periods=years * 365, freq="D")
    forecast = model.predict(future)
    return model, forecast


def compute_annual_cagr(start_val: float, end_val: float, years: float) -> float:
    if start_val <= 0 or years <= 0:
        return 0.0
    return (end_val / start_val) ** (1.0 / years) - 1.0


def plot_and_encode(df_hist: pd.DataFrame, forecast: pd.DataFrame, title: str):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_hist["ds"], df_hist["y"], label="Historical", color="blue")
    ax.plot(forecast["ds"], forecast["yhat"], label="Forecast", color="orange")
    ax.fill_between(forecast["ds"], forecast["yhat_lower"], forecast["yhat_upper"], color="orange", alpha=0.2)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    img_bytes = buf.read()
    return img_bytes, base64.b64encode(img_bytes).decode("utf-8")


def call_azure_gpt(system_prompt: str, user_prompt: str) -> str:
    if not (AZURE_KEY and AZURE_ENDPOINT and AZURE_DEPLOYMENT):
        raise RuntimeError("Azure OpenAI not configured in .env")
    url = f"{AZURE_ENDPOINT}openai/deployments/{AZURE_DEPLOYMENT}/chat/completions?api-version=2024-02-15-preview"
    headers = {"Content-Type": "application/json", "api-key": AZURE_KEY}
    body = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": 1200,
        "temperature": 0.6
    }
    r = requests.post(url, headers=headers, json=body, timeout=60)
    r.raise_for_status()
    j = r.json()
    try:
        return j["choices"][0]["message"]["content"]
    except Exception:
        return json.dumps(j)


@app.route("/idea", methods=["POST"])
def idea_evaluate():
    try:
        payload = request.get_json() or {}
        idea = payload.get("idea", "").strip()
        starting_cap = float(payload.get("starting_capital") or payload.get("funding") or 0.0)
        forecast_years = int(payload.get("forecast_years", 5))
        sector = payload.get("sector") or None

        if not idea:
            return jsonify({"error": "Provide 'idea' text"}), 400

        ticker = map_idea_to_ticker(sector if sector else idea)

        series = fetch_close_series(ticker, period="10y")
        df = prepare_df(series)
        model, forecast = fit_forecast(df, years=forecast_years)

        last_hist_val = float(df["y"].iloc[-1])
        last_hist_date = df["ds"].iloc[-1]
        future_part = forecast[forecast["ds"] > last_hist_date].reset_index(drop=True)
        if future_part.empty:
            future_part = forecast

        final_yhat = float(future_part["yhat"].iloc[-1])
        years_span = (future_part["ds"].iloc[-1] - future_part["ds"].iloc[0]).days / 365.25 or forecast_years

        forecast_cagr = compute_annual_cagr(last_hist_val, final_yhat, years_span)
        trend = "Increase" if forecast_cagr > 0.02 else ("Decrease" if forecast_cagr < -0.02 else "Stable")
        hist_vol = series.pct_change().dropna().std() if len(series) > 10 else 0.0
        conf_raw = min(1.0, max(0.0, abs(forecast_cagr) * 5)) * (1.0 - min(0.9, hist_vol * 5))
        confidence = round(conf_raw * 100, 1)
        projected_value = starting_cap * ((1.0 + forecast_cagr) ** forecast_years) if starting_cap > 0 else 0.0
        projected_profit = projected_value - starting_cap

        chart_bytes, chart_b64 = plot_and_encode(df, forecast, title=f"{ticker} historical + {forecast_years}y forecast")

        # Build JSON-prompt for Azure (strict JSON output)
        system_prompt = (
            "You are an expert startup advisor and market analyst. "
            "Return valid JSON only (no surrounding explanation)."
        )

        # The desired JSON schema we ask for
        user_prompt = (
            "Produce a JSON object with these fields:\n"
            " {\n"
            "  \"summary\": string,\n"
            "  \"key_experiences\": [string],\n"
            "  \"top_challenges\": [string],\n"
            "  \"common_mistakes\": [string],\n"
            "  \"recommendations\": [string],\n"
            "  \"competitors\": [{\"name\": string, \"why\": string, \"difference_from_idea\": string}],\n"
            "  \"swot\": {\"strengths\": [string], \"weaknesses\": [string], \"opportunities\": [string], \"threats\": [string]}\n"
            " }\n\n"
            f"Context:\n idea: {idea}\n sector_proxy: {ticker}\n starting_capital: {starting_cap}\n forecast_cagr: {forecast_cagr:.4f}\n trend: {trend}\n confidence: {confidence}%\n\n"
            "Keep entries short (1-2 sentences each). Return JSON only."
        )

        ai_raw = call_azure_gpt(system_prompt, user_prompt)

        # Azure may return a JSON string or may include extra text; try to parse robustly:
        ai_struct = None
        try:
            ai_struct = json.loads(ai_raw)
        except Exception:
            # Attempt to extract the first JSON substring
            import re
            m = re.search(r"\{.*\}", ai_raw, flags=re.DOTALL)
            if m:
                try:
                    ai_struct = json.loads(m.group(0))
                except Exception:
                    ai_struct = {"error_parsing_ai": ai_raw}
            else:
                ai_struct = {"error_parsing_ai": ai_raw}

        resp = {
            "trend": trend,
            "confidence": confidence,
            "forecast_cagr": forecast_cagr,
            "projected_value": projected_value,
            "projected_profit": projected_profit,
            "ticker": ticker,
            "chart": chart_b64,
            "ai": ai_struct,
            "metrics": {
                "last_hist_value": last_hist_val,
                "final_forecast_value": final_yhat,
                "forecast_years": forecast_years,
                "hist_volatility": hist_vol
            }
        }

        return jsonify(resp)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5002)), debug=True)
