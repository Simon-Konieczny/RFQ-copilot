"""The tools the LLM can call: their declarations and implementations.

The model only passes the RFQ as the client stated it (strike 105 in "pct_spot",
tenor "3M"). Every number is computed in code, inside tools.price_rfq.

There is deliberately no tool to send a quote: approval happens outside the
LLM loop, so no message can talk the model into skipping it.

Every tool has the same signature, fn(args: dict, trace: list) -> dict, so the
loop can dispatch them uniformly. Bad arguments raise (pydantic.ValidationError
or ValueError); turning that into an error result for the model is the loop's job.
"""
from typing import Literal, get_args, get_origin

from rfq_copilot.schema import Expiry, RFQModel, Strike
from rfq_copilot.tools import price_rfq, validate_rfq


def _enum(model, field: str) -> list[str]:
    """Allowed values of a schema field, read from its Literal type, so the
    declaration can't drift from the schema. get_args unwraps one layer:
    Literal[...] | None -> (Literal[...], None); list[Literal[...]] -> (Literal[...],)."""
    for arg in get_args(model.model_fields[field].annotation):
        if get_origin(arg) is Literal:
            return list(get_args(arg))
    raise TypeError(f"{field} is not a Literal field")


# Flat on purpose
RFQ_PARAMETERS = {
    "type": "object",
    "properties": {
        "request_type": {"type": "string", "enum": _enum(RFQModel, "request_type"),
                         "description": "rfq: a single vanilla call/put request. not_rfq: chat, "
                                        "research, or anything that isn't a price request. "
                                        "unsupported: a price request we can't auto-quote."},
        "unsupported_reason": {"type": "string", "enum": _enum(RFQModel, "unsupported_reason"),
                               "description": "Only when request_type is unsupported."},
        "side": {"type": "string", "enum": _enum(RFQModel, "side"),
                 "description": "From the client's point of view. Omit if not stated."},
        "option_type": {"type": "string", "enum": _enum(RFQModel, "option_type")},
        "underlying": {"type": "string",
                       "description": "As the client wrote it, e.g. 'Euro Stoxx 50' or 'SX5E'."},
        "strike_unit": {"type": "string", "enum": _enum(Strike, "unit")},
        "strike_value": {"type": "number",
                         "description": "As stated: 105 for 105%, 5500 for an absolute strike. "
                                        "Omit for atm/atmf. Never compute it."},
        "expiry_kind": {"type": "string", "enum": _enum(Expiry, "kind")},
        "expiry_value": {"type": "string",
                         "description": "tenor like '3M', '1Y', '2W'; month like '2027-03'; "
                                        "or ISO date like '2027-06-18'."},
        "notional": {"type": "number", "description": "In currency units: 10mm -> 10000000."},
        "currency": {"type": "string", "description": "3-letter code. Omit if not stated."},
        "ambiguities": {"type": "array", "items": {"type": "string", "enum": _enum(RFQModel, "ambiguities")},
                        "description": "Flag instead of guessing when the text allows two readings."},
    },
    "required": ["request_type"],
}


def rfq_from_args(args: dict) -> RFQModel:
    """Flat tool arguments -> RFQModel. Omitted fields mean 'not stated' (None)."""
    unknown = set(args) - set(RFQ_PARAMETERS["properties"])
    if unknown:
        # RFQModel would silently drop these; a misspelt field must not vanish.
        raise ValueError(f"unknown arguments: {sorted(unknown)}")
    a = dict(args)
    strike = {"unit": a.pop("strike_unit", None), "value": a.pop("strike_value", None)}
    expiry = {"kind": a.pop("expiry_kind", None), "value": a.pop("expiry_value", None)}
    return RFQModel(
        **a,
        strike=None if all(v is None for v in strike.values()) else strike, # type: ignore
        expiry=None if all(v is None for v in expiry.values()) else expiry, # type: ignore
    )


def validate_rfq_tool(args: dict, trace: list) -> dict:
    return validate_rfq(rfq_from_args(args))


def price_rfq_tool(args: dict, trace: list) -> dict:
    return price_rfq(rfq_from_args(args), trace) # type: ignore


TOOLS = {
    "validate_rfq": validate_rfq_tool,
    "price_rfq": price_rfq_tool,
}

# Provider-neutral declarations; llm.py converts them to the SDK's format.
TOOL_DECLARATIONS = [
    {"name": "validate_rfq",
     "description": "Submit the RFQ exactly as the client stated it. Returns the action "
                    "(price | clarify | reject) and the reasons. Call this first.",
     "parameters": RFQ_PARAMETERS},
    {"name": "price_rfq",
     "description": "Price an RFQ that validate_rfq accepted (same arguments). Returns the "
                    "price, premium, delta and vega computed by the bank's pricer. These are "
                    "the only numbers you may quote.",
     "parameters": RFQ_PARAMETERS},
]
