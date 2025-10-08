"""
Huawei Cloud DLI Flink job helper.

This module wraps the Huawei Cloud Python SDK so Streamlit pages can
submit and manage Flink SQL jobs with concise, well-typed helpers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from huaweicloudsdkcore.auth.credentials import BasicCredentials
from huaweicloudsdkcore.exceptions import exceptions as sdk_exceptions
from huaweicloudsdkdli.v1 import (
    BatchRunFlinkJobsRequest,
    BatchRunFlinkJobsRequestBody,
    BatchStopFlinkJobsRequest,
    CreateFlinkJarJobRequest,
    CreateFlinkJarJobRequestBody,
    CreateFlinkSqlJobRequest,
    CreateFlinkSqlJobRequestBody,
    DliClient,
    ListFlinkJobsRequest,
    ShowFlinkJobRequest,
    StopFlinkJobsRequestBody,
)
from huaweicloudsdkdli.v1.region.dli_region import DliRegion

LOG = logging.getLogger(__name__)


class DLIClientError(RuntimeError):
    """Raised when the Huawei Cloud SDK reports an error."""


@dataclass
class FlinkJobHandle:
    """Minimal information to reference a Flink job."""

    job_id: str
    job_type: Optional[str] = None
    status: Optional[str] = None
    status_desc: Optional[str] = None


def _compact_dict(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Remove None values so optional SDK fields remain unset."""
    return {key: value for key, value in payload.items() if value is not None}


class DLIClient:
    """
    Thin wrapper around the Huawei Cloud DLI SDK with helpful defaults.
    """

    def __init__(
        self,
        ak: str,
        sk: str,
        project_id: str,
        region: str,
        *,
        endpoint: Optional[str] = None,
    ) -> None:
        if not all([ak, sk, project_id, region]):
            raise DLIClientError("AK, SK, project_id, and region must all be provided.")

        credentials = BasicCredentials(ak=ak, sk=sk, project_id=project_id)

        builder = DliClient.new_builder().with_credentials(credentials)
        if endpoint:
            builder = builder.with_endpoint(endpoint)
        else:
            try:
                builder = builder.with_region(DliRegion.value_of(region))
            except KeyError as exc:
                raise DLIClientError(f"Unknown Huawei Cloud region: {region}") from exc

        self._client = builder.build()

    def submit_flink_sql_job(
        self,
        *,
        name: str,
        queue_name: str,
        sql_body: str,
        run_mode: str = "shared_cluster",
        cu_number: int = 2,
        parallel_number: Optional[int] = None,
        checkpoint_enabled: bool = False,
        checkpoint_interval: Optional[int] = None,
        log_enabled: bool = True,
        obs_bucket: Optional[str] = None,
        resume_checkpoint: Optional[bool] = None,
        runtime_config: Optional[str] = None,
        flink_version: Optional[str] = None,
        execution_agency_urn: Optional[str] = None,
    ) -> FlinkJobHandle:
        """
        Create a Flink SQL job definition in DLI.
        """
        body = CreateFlinkSqlJobRequestBody(
            **_compact_dict(
                {
                    "name": name,
                    "queue_name": queue_name,
                    "sql_body": sql_body,
                    "run_mode": run_mode,
                    "cu_number": cu_number,
                    "parallel_number": parallel_number,
                    "checkpoint_enabled": checkpoint_enabled,
                    "checkpoint_interval": checkpoint_interval,
                    "log_enabled": log_enabled,
                    "obs_bucket": obs_bucket,
                    "resume_checkpoint": resume_checkpoint,
                    "runtime_config": runtime_config,
                    "flink_version": flink_version,
                    "execution_agency_urn": execution_agency_urn,
                }
            )
        )
        request = CreateFlinkSqlJobRequest(body=body)

        try:
            response = self._client.create_flink_sql_job(request)
        except sdk_exceptions.ClientRequestException as sdk_error:
            LOG.exception("Failed to create Flink SQL job via DLI.")
            raise DLIClientError(
                f"创建 Flink SQL 作业失败: {sdk_error.error_msg}"
            ) from sdk_error

        if not getattr(response, "job", None):
            raise DLIClientError("DLI 未返回作业信息。")

        job_id = str(response.job.job_id)
        status = getattr(response.job, "status_name", None)
        status_desc = getattr(response.job, "status_desc", None)

        return FlinkJobHandle(
            job_id=job_id,
            job_type="flink_sql_job",
            status=status,
            status_desc=status_desc,
        )

    def submit_flink_jar_job(
        self,
        *,
        name: str,
        queue_name: str,
        cu_number: int,
        manager_cu_number: int,
        tm_cus: Optional[int],
        tm_slot_num: Optional[int],
        parallel_number: Optional[int],
        log_enabled: bool,
        obs_bucket: Optional[str],
        main_class: str,
        entrypoint: str,
        entrypoint_args: Optional[str],
        dependency_jars: Optional[List[str]],
        dependency_files: Optional[List[str]],
        restart_when_exception: bool,
        flink_version: Optional[str],
        runtime_config: Optional[str],
        execution_agency_urn: Optional[str],
        resume_checkpoint: Optional[bool],
        resume_max_num: Optional[int],
        checkpoint_path: Optional[str],
    ) -> FlinkJobHandle:
        """
        Create a Flink custom (JAR) job definition in DLI.
        """
        payload = _compact_dict(
            {
                "name": name,
                "queue_name": queue_name,
                "cu_number": cu_number,
                "manager_cu_number": manager_cu_number,
                "parallel_number": parallel_number,
                "log_enabled": log_enabled,
                "obs_bucket": obs_bucket,
                "main_class": main_class,
                "entrypoint": entrypoint,
                "entrypoint_args": entrypoint_args,
                "dependency_jars": dependency_jars,
                "dependency_files": dependency_files,
                "restart_when_exception": restart_when_exception,
                "tm_cus": tm_cus,
                "tm_slot_num": tm_slot_num,
                "flink_version": flink_version,
                "runtime_config": runtime_config,
                "execution_agency_urn": execution_agency_urn,
                "resume_checkpoint": resume_checkpoint,
                "resume_max_num": resume_max_num,
                "checkpoint_path": checkpoint_path,
            }
        )

        body = CreateFlinkJarJobRequestBody(**payload)
        request = CreateFlinkJarJobRequest(body=body)

        try:
            response = self._client.create_flink_jar_job(request)
        except sdk_exceptions.ClientRequestException as sdk_error:
            LOG.exception("Failed to create Flink Jar job via DLI.")
            raise DLIClientError(
                f"创建 Flink 自定义作业失败: {sdk_error.error_msg}"
            ) from sdk_error

        if not getattr(response, "job", None):
            raise DLIClientError("DLI 未返回自定义作业信息。")

        job_id = str(response.job.job_id)
        status = getattr(response.job, "status_name", None)
        status_desc = getattr(response.job, "status_desc", None)

        return FlinkJobHandle(
            job_id=job_id,
            job_type="flink_jar_job",
            status=status,
            status_desc=status_desc,
        )

    def run_flink_job(self, job_id: str, resume_savepoint: Optional[bool] = None) -> None:
        """
        Trigger execution for an existing Flink job.
        """
        body = BatchRunFlinkJobsRequestBody(
            **_compact_dict({"job_ids": [job_id], "resume_savepoint": resume_savepoint})
        )
        request = BatchRunFlinkJobsRequest(body=body)

        try:
            response = self._client.batch_run_flink_jobs(request)
        except sdk_exceptions.ClientRequestException as sdk_error:
            LOG.exception("Failed to start Flink job.")
            raise DLIClientError(f"启动 Flink 作业失败: {sdk_error.error_msg}") from sdk_error

        for result in getattr(response, "body", []) or []:
            if getattr(result, "is_success", True):
                continue
            raise DLIClientError(f"启动作业失败: {getattr(result, 'message', '未知错误')}")

    def stop_flink_job(
        self, job_id: str, *, trigger_savepoint: Optional[bool] = None
    ) -> None:
        """
        Stop a running Flink job.
        """
        body = StopFlinkJobsRequestBody(
            **_compact_dict({"job_ids": [job_id], "trigger_savepoint": trigger_savepoint})
        )
        request = BatchStopFlinkJobsRequest(body=body)

        try:
            response = self._client.batch_stop_flink_jobs(request)
        except sdk_exceptions.ClientRequestException as sdk_error:
            LOG.exception("Failed to stop Flink job.")
            raise DLIClientError(f"停止 Flink 作业失败: {sdk_error.error_msg}") from sdk_error

        for result in getattr(response, "body", []) or []:
            if getattr(result, "is_success", True):
                continue
            raise DLIClientError(f"停止作业失败: {getattr(result, 'message', '未知错误')}")

    def get_job_detail(self, job_id: str) -> Dict[str, Any]:
        """
        Fetch detailed status for a specific job.
        """
        request = ShowFlinkJobRequest(job_id=job_id)
        try:
            response = self._client.show_flink_job(request)
        except sdk_exceptions.ClientRequestException as sdk_error:
            LOG.exception("Failed to query Flink job detail.")
            raise DLIClientError(f"查询作业详情失败: {sdk_error.error_msg}") from sdk_error

        job = getattr(response, "job_detail", None)
        if not job:
            raise DLIClientError("未找到指定的作业。")

        job_dict: Dict[str, Any] = {
            "job_id": str(getattr(job, "job_id", job_id)),
            "name": getattr(job, "name", ""),
            "job_type": getattr(job, "job_type", ""),
            "status": getattr(job, "status", ""),
            "status_desc": getattr(job, "status_desc", ""),
            "queue_name": getattr(job, "queue_name", ""),
            "sql_body": getattr(job, "sql_body", ""),
            "create_time": getattr(job, "create_time", None),
            "start_time": getattr(job, "start_time", None),
            "duration": getattr(job, "duration", None),
            "restart_times": getattr(job, "restart_times", None),
        }

        config = getattr(job, "job_config", None)
        if config:
            job_dict["config"] = config.to_dict()  # type: ignore[attr-defined]

        return job_dict

    def list_jobs(
        self,
        *,
        status: Optional[str] = None,
        queue_name: Optional[str] = None,
        limit: int = 50,
        show_detail: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        List recent Flink jobs with optional filters.
        """
        request = ListFlinkJobsRequest(
            **_compact_dict(
                {
                    "status": status,
                    "queue_name": queue_name,
                    "limit": limit,
                    "show_detail": show_detail,
                }
            )
        )

        try:
            response = self._client.list_flink_jobs(request)
        except sdk_exceptions.ClientRequestException as sdk_error:
            LOG.exception("Failed to list Flink jobs.")
            raise DLIClientError(f"获取作业列表失败: {sdk_error.error_msg}") from sdk_error

        jobs: List[Dict[str, Any]] = []
        for job in getattr(response, "jobs", []) or []:
            jobs.append(
                {
                    "job_id": str(getattr(job, "job_id", "")),
                    "name": getattr(job, "name", ""),
                    "job_type": getattr(job, "job_type", ""),
                    "status": getattr(job, "status_name", ""),
                    "status_desc": getattr(job, "status_desc", ""),
                    "queue_name": getattr(job, "queue_name", ""),
                    "create_time": getattr(job, "create_time", None),
                    "update_time": getattr(job, "update_time", None),
                }
            )

        return jobs
