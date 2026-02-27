## 2024-03-22 - [Hardcoded Configuration]
**Vulnerability:** The spreadsheet ID and credential file path were hardcoded in `lotto_predict.py`, potentially exposing sensitive configuration details.
**Learning:** Hardcoding such values makes it difficult to change environments (dev vs. prod) and increases the risk of accidental exposure if the code is shared publicly.
**Prevention:** Use environment variables for all configuration settings, providing defaults only when safe and appropriate.
