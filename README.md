# Scalable Statistical EDA Engine (Phase 1–3)

## Overview
This project provides a FastAPI-based statistical EDA engine that:
- Infers problem type (classification vs regression)
- Extracts cheap but powerful statistical signals
- Scores and filters model families
- Scales safely to large datasets

## Run
```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

## Endpoint
POST `/eda/phase1-3`
- file: CSV
- target_column: name of target

## Future Scope
- Phase 4: Auto feature engineering
- Phase 5: Confidence-weighted model ranking
- Phase 6: Supervising agent (adaptive tuning)