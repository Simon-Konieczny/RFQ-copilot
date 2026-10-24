import pandas as pd
from pathlib import Path
import re
from datetime import date, timedelta
from math import sqrt, log, exp
from scipy.stats import norm

from rfq_copilot.schema import RFQModel, Expiry

# src/rfq_copilot/tools.py -> parents[2] is the repo root
DATA = Path(__file__).resolve().parents[2] / 'data'
market_df = pd.read_csv(DATA / 'market.csv')

alias_lookup = {}

for idx, row in enumerate(market_df.itertuples(index=False)):
    if pd.notna(row.ticker):
        alias_lookup[str(row.ticker).lower().strip()] = idx
    
    if pd.notna(row.aliases):
        for alias in str(row.aliases).lower().split('|'):
            alias_lookup[alias.strip()] = idx

MAX_TENOR_YEARS = 10

def get_market_data(underlying) -> dict | None:
    u = underlying.strip().lower()
    target_cols = ['ticker', 'as_of', 'currency', 'spot', 'vol', 'rate', 'div_yield']

    idx = alias_lookup.get(u)
    if (idx is not None):
        match = market_df.iloc[idx][target_cols]
        return match.to_dict()

    return None

def price_option(S, K, T, r, q, sigma, option_type) -> dict | None:
    if (option_type not in ["call", "put"]):
        raise ValueError("option_type must be 'call' or 'put'")


    d1 = (log(S / K) + (r - q + sigma**2/2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)

    vega = S * exp((-q * T)) * norm.pdf(d1) * sqrt(T)

    if (option_type == "call"):
        call = S * exp(-q * T) * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)
        delta = exp(-q * T) * norm.cdf(d1)
        
        return {"price": call, "delta": delta, "vega": vega}
    
    elif (option_type == "put"):
        put = K * exp(-r * T) * norm.cdf(-d2) - S * exp(-q * T) * norm.cdf(-d1)
        delta = -exp(-q * T) * norm.cdf(-d1)
        
        return {"price": put, "delta": delta, "vega": vega}

    return None


def third_friday(year: int, month: int) -> date:
    d = date(year, month, 1)
    d += timedelta(days=(4 - d.weekday()) % 7)  # first Friday (Mon=0 ... Fri=4)
    return d + timedelta(days=14)


def year_fraction(expiry: Expiry, as_of: date) -> float:
    assert expiry.value is not None
    v = expiry.value.strip().upper()
    
    if expiry.kind == "tenor":
        m = re.fullmatch(r"(\d+)([WMY])", v)
        if not m:
            raise ValueError(f"bad tenor: {expiry.value}")
        n, unit = int(m.group(1)), m.group(2)
        return {"W": 7 * n / 365, "M": n / 12, "Y": float(n)}[unit]
    if expiry.kind == "month":
        y, mth = map(int, v.split("-"))
        d = third_friday(y, mth)
    else:
        d = date.fromisoformat(v)
    return (d - as_of).days / 365


def validate_rfq(rfq: RFQModel) -> dict:
    # 1. Reject

    if rfq.request_type == "not_rfq":
        return _get_message("reject", ["not_an_rfq"])
    if rfq.request_type != "rfq":
        return _get_message("reject", [rfq.unsupported_reason or "unsupported_request"])

    mkt = None
    if rfq.underlying:
        mkt = get_market_data(rfq.underlying)
        if mkt is None:
            return _get_message("reject", ["unknown_underlying"])

    # 2. Clarify: missing fields. Side and currency are optional

    reasons = []
    if not rfq.underlying:
        reasons.append("missing_underlying")
    if rfq.option_type is None:
        reasons.append("missing_option_type")

    # ATM/ATMF strikes have no value; only abs and pct_spot need one.
    if rfq.strike is None or rfq.strike.unit is None:
        reasons.append("missing_strike")
    elif rfq.strike.unit in ("abs", "pct_spot"):
        if rfq.strike.value is None:
            reasons.append("missing_strike")
        elif rfq.strike.value <= 0:
            reasons.append("invalid_strike")

    if rfq.notional is None:
        reasons.append("missing_notional")
    elif rfq.notional <= 0:
        reasons.append("invalid_notional")

    # 3. Clarify: invalid values.

    if rfq.expiry is None or rfq.expiry.kind is None or not rfq.expiry.value:
        reasons.append("missing_expiry")
    else:
        as_of = date.fromisoformat(mkt["as_of"]) if mkt else date.today()
        try:
            T = year_fraction(rfq.expiry, as_of)
        except ValueError:
            reasons.append("invalid_expiry")
        else:
            if T <= 0:
                reasons.append("expiry_in_past")
            elif T > MAX_TENOR_YEARS:
                reasons.append("tenor_out_of_range")

    if mkt and rfq.currency and rfq.currency.strip().upper() != mkt["currency"]:
        reasons.append("currency_mismatch")

    # 4. Clarify: ambiguities only the LLM can spot (e.g. "3m" = months or millions?)
    reasons.extend(rfq.ambiguities)

    if reasons:
        return _get_message("clarify", reasons)
    
    return _get_message("price", [])


def _get_message(action: str, reasons: list[str]) -> dict:
    return {"action": action, "reasons": reasons}
