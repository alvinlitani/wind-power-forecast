"""Prefect flows for the wind power forecasting pipeline.

Three independent flows, each schedulable on its own:
    - ingest_flow   : IESO actuals download + preprocess
    - predict_flow  : weather fetch + LSTM and/or XGBoost prediction
    - evaluate_flow : evaluate yesterday's predictions vs IESO actuals
"""
