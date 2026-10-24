from pydantic import BaseModel, Field
from typing import Literal

class Expiry(BaseModel):
    kind: Literal["tenor", "month", "date"] | None = None
    value: str | None = Field(None, description='tenor like "3M"/"1Y"/"2W", month "2026-12", or ISO date')

class Strike(BaseModel):
    unit: Literal["abs","pct_spot","atm","atmf"] | None = None
    value: float | None = Field(None, description='105 for 105% (pct_spot), 5670 for an absolute strike; None for atm/atmf')

class RFQModel(BaseModel):
    side: Literal["buy", "sell", "two_way"] | None = "two_way"
    option_type: Literal["call", "put"] | None = None
    underlying: str | None = Field(None, description='ticker like "DAX" or "SPX"')
    strike: Strike | None = None
    expiry: Expiry | None = None
    notional: float | None = None
    currency: str | None = Field(None, description='3 letter currency symbol like "EUR"')
    request_type: Literal["rfq", "not_rfq", "unsupported"] | None = None
    # Only set when request_type == "unsupported": says why, so the reject reason can be scored.
    unsupported_reason: Literal["multi_leg_unsupported", "unsupported_product", "multiple_requests",
                                "unsupported_strike_convention", "advisory_request"] | None = None
    ambiguities: list[Literal["ambiguous_3m_tenor_or_size", "ambiguous_strike_unit", "ambiguous_notional_unit",
                              "missing_context_reference"]] = Field(default_factory=list)