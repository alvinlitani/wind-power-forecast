from prefect import flow
from prefect.runner.storage import GitRepository

SOURCE = GitRepository(url="https://github.com/alvinlitani/wind-power-forecast.git")
ENTRYPOINT = "orchestration/trigger.py:trigger_wind_forecast_job"

if __name__ == "__main__":
    flow.from_source(source=SOURCE, entrypoint=ENTRYPOINT).deploy(
        name="daily-trigger",
        work_pool_name="wf-managed",
        job_variables={"pip_packages": ["google-cloud-run"]},
    )