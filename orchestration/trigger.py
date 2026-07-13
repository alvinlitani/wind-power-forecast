"""Prefect control-plane flow: trigger the wind-forecast-job Cloud Run Job.

Fire-and-forget under Architecture T. This flow starts the Job's execution and
returns immediately -- it does NOT wait for the Job to finish. Trigger-time
failures (auth, job-not-found, quota) surface here as a failed Prefect run;
runtime failures (fetch / roster gate / predict) live in Cloud Run's own
execution history.

Auth: loads the invoke-only SA key from the Prefect Secret block and builds a
google credential at runtime. The two GCP calls below map exactly onto the two
permissions in the wfJobInvoker custom role (run.jobs.get, run.jobs.run).
"""

import json

from prefect import flow, get_run_logger
from prefect.artifacts import create_markdown_artifact
from prefect.blocks.system import Secret

from google.cloud import run_v2
from google.oauth2 import service_account

PROJECT = "wind-forecast-ontario"
REGION = "us-central1"
JOB = "wind-forecast-job"
JOB_NAME = f"projects/{PROJECT}/locations/{REGION}/jobs/{JOB}"
SECRET_BLOCK = "wf-job-invoker-key"


def _build_client() -> run_v2.JobsClient:
    key = Secret.load(SECRET_BLOCK).get()
    info = key if isinstance(key, dict) else json.loads(key)
    creds = service_account.Credentials.from_service_account_info(info)
    return run_v2.JobsClient(credentials=creds)


@flow(name="trigger-wind-forecast-job")
def trigger_wind_forecast_job():
    logger = get_run_logger()
    client = _build_client()

    # run.jobs.get -- read the code version the Job is configured to run.
    job = client.get_job(name=JOB_NAME)
    env = {e.name: e.value for c in job.template.template.containers for e in c.env}
    code_sha = env.get("CODE_SHA", "local")

    # run.jobs.run -- start the execution. Returns an operation immediately;
    # we deliberately do NOT call .result(), so this stays fire-and-forget.
    operation = client.run_job(name=JOB_NAME)
    execution_name = getattr(operation.metadata, "name", None)

    logger.info(
        "Triggered %s (code_sha=%s); execution=%s", JOB, code_sha, execution_name
    )

    create_markdown_artifact(
        key="job-trigger",
        markdown=(
            f"## wind-forecast-job triggered\n\n"
            f"- **code_sha:** `{code_sha}`\n"
            f"- **execution:** `{execution_name}`\n"
            f"- **mode:** fire-and-forget "
            f"(runtime outcome in Cloud Run execution history)\n"
        ),
    )

    return {"code_sha": code_sha, "execution": execution_name}


if __name__ == "__main__":
    trigger_wind_forecast_job()
