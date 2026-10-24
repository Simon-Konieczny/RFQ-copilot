# RFQ-copilot

## Labelling Conventions
This section outlines the rules for dealing with abigiuities and labelling conventions.
- `side` is always from the clients perspective: "your bid" means the client sells.
- if the currency is missing, it is autofilled with the underlying's. in case of a mismatch, a clarification is requested
- a month-only expiry means the third Friday of that month
- Max tenor is 10Y
- Multi-leg trades are auto-rejected, not supported.
- After two clarifications, the trade is rejected.