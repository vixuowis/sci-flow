from __future__ import annotations

import os
from typing import List, Optional

import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

from dli_client import DLIClient, DLIClientError
from integration_helper import add_auth_sidebar, check_authentication, require_role

RUN_MODE_DESCRIPTIONS = {
    "shared_cluster": "共享资源池：与其他作业共用 DLI 计算资源，适合测试或轻量级任务，按使用量计费。",
    "exclusive_cluster": "独享资源池：独立租用专属集群，隔离性和性能稳定性更好，适合生产关键业务。",
    "edge_node": "边缘节点：在 IEF 等边缘环境中运行作业，需要已配置的边缘节点资源。",
}


@st.cache_resource(show_spinner=False)
def build_dli_client(
    ak: str,
    sk: str,
    project_id: str,
    region: str,
    endpoint: Optional[str],
) -> DLIClient:
    return DLIClient(ak=ak, sk=sk, project_id=project_id, region=region, endpoint=endpoint)


def _parse_optional_int(raw: str) -> Optional[int]:
    text = (raw or "").strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _parse_resource_list(raw: str) -> Optional[List[str]]:
    if not raw:
        return None
    items = []
    for token in raw.replace(",", "\n").splitlines():
        token = token.strip()
        if token:
            items.append(token)
    return items or None


def load_default_config() -> dict:
    secrets_cfg = {}
    if hasattr(st, "secrets"):
        try:
            secrets_cfg = st.secrets.get("dli", {})
        except (StreamlitSecretNotFoundError, AttributeError, KeyError):
            secrets_cfg = {}
    return {
        "ak": secrets_cfg.get("ak") or os.environ.get("DLI_AK"),
        "sk": secrets_cfg.get("sk") or os.environ.get("DLI_SK"),
        "project_id": secrets_cfg.get("project_id") or os.environ.get("DLI_PROJECT_ID"),
        "region": secrets_cfg.get("region") or os.environ.get("DLI_REGION", "cn-north-4"),
        "queue_name": secrets_cfg.get("queue_name") or os.environ.get("DLI_QUEUE_NAME", "default"),
        "endpoint": secrets_cfg.get("endpoint") or os.environ.get("DLI_ENDPOINT"),
    }


def main() -> None:
    user_info = check_authentication()
    require_role("researcher")
    add_auth_sidebar()

    st.title("Flink 作业中心")
    st.caption("在华为云 DLI 中提交、启动与监控 Flink SQL 与自定义 Jar 作业。")

    defaults = load_default_config()

    with st.expander("连接配置", expanded=True):
        st.markdown("请填写华为云 DLI 访问参数，建议通过环境变量或 `st.secrets` 注入。")
        ak = st.text_input("Access Key (AK)", value=st.session_state.get("dli_ak", defaults["ak"]) or "")
        sk = st.text_input(
            "Secret Key (SK)",
            value=st.session_state.get("dli_sk", defaults["sk"]) or "",
            type="password",
            help="建议使用环境变量或 Streamlit secrets 管理凭据。",
        )
        project_id = st.text_input(
            "Project ID",
            value=st.session_state.get("dli_project_id", defaults["project_id"]) or "",
        )
        region = st.text_input(
            "Region",
            value=st.session_state.get("dli_region", defaults["region"]) or "",
            help="例如：cn-north-4。",
        )
        queue_name = st.text_input(
            "Queue 名称",
            value=st.session_state.get("dli_queue_name", defaults["queue_name"]) or "",
        )
        endpoint = st.text_input(
            "自定义 Endpoint（可选）",
            value=st.session_state.get("dli_endpoint", defaults["endpoint"]) or "",
            help="如需自定义访问地址，可填写完整 HTTPS endpoint。",
        )

        st.session_state["dli_ak"] = ak
        st.session_state["dli_sk"] = sk
        st.session_state["dli_project_id"] = project_id
        st.session_state["dli_region"] = region
        st.session_state["dli_queue_name"] = queue_name
        st.session_state["dli_endpoint"] = endpoint or None

    missing_fields = [label for label, value in [
        ("AK", ak),
        ("SK", sk),
        ("Project ID", project_id),
        ("Region", region),
        ("Queue 名称", queue_name),
    ] if not value]

    if missing_fields:
        st.warning(f"请完善连接配置：{', '.join(missing_fields)}")
        st.stop()

    try:
        client = build_dli_client(ak, sk, project_id, region, endpoint or None)
    except DLIClientError as error:
        st.error(f"创建 DLI 客户端失败：{error}")
        st.stop()

    st.subheader("提交作业")
    with st.form("flink_job_form"):
        job_variant = st.radio(
            "作业类型",
            options=("Flink SQL", "自定义 Jar"),
            index=0 if st.session_state.get("dli_job_variant", "sql") == "sql" else 1,
            horizontal=True,
        )
        job_type = "sql" if job_variant == "Flink SQL" else "jar"

        col_common_1, col_common_2 = st.columns(2)
        with col_common_1:
            job_name = st.text_input("作业名称", value=st.session_state.get("dli_job_name", "sci_flow_job"))
            cu_number = st.number_input(
                "CU 数量",
                min_value=1,
                step=1,
                value=st.session_state.get("dli_cu_number", 2),
            )
            parallel_number = st.number_input(
                "并行度 (可选)",
                min_value=1,
                step=1,
                value=st.session_state.get("dli_parallel_number", 1),
            )
        with col_common_2:
            log_enabled = st.checkbox(
                "启用作业日志 (OBS)",
                value=st.session_state.get("dli_log_enabled", False),
                help="启用后，Flink 作业的日志会写入指定 OBS 桶，需要具备相应权限。",
            )
            auto_run = st.checkbox(
                "提交后立即启动作业",
                value=st.session_state.get("dli_auto_run", True),
            )

        obs_bucket: Optional[str] = None
        if log_enabled:
            obs_bucket = st.text_input(
                "OBS 桶名称（启用日志必填）",
                value=st.session_state.get("dli_obs_bucket", ""),
                help="填写 OBS 桶名，例如 my-dli-logs；需要与项目同区域并已授权。",
            )

        execution_agency_urn = st.text_input(
            "执行委托 URN（需访问 OBS/RDS 等资源时必填）",
            value=st.session_state.get("dli_execution_agency", ""),
            help="在 IAM > 委托 中创建“云服务”类型的委托，选择 DLI 作为受信任服务，授予对 OBS/RDS 等资源的权限，然后复制 URN。",
        )

        flink_version = st.text_input(
            "Flink 版本（可选）",
            value=st.session_state.get("dli_flink_version", ""),
            help="示例：1.15。未填写时采用队列默认版本。",
        )

        runtime_config = st.text_area(
            "高级运行配置 JSON（可选）",
            value=st.session_state.get("dli_runtime_config", ""),
            height=120,
            help="示例：{\"state.backend\": \"filesystem\"}",
        )

        sql_body = ""
        run_mode = "exclusive_cluster"
        checkpoint_enabled = False
        checkpoint_interval = 60000

        manager_cu_number = st.session_state.get("dli_manager_cu_number", 1)
        tm_cus_input = st.session_state.get("dli_tm_cus_text", "")
        tm_slot_input = st.session_state.get("dli_tm_slot_text", "")
        restart_when_exception = st.session_state.get("dli_restart_when_exception", True)
        main_class = st.session_state.get("dli_main_class", "")
        entrypoint = st.session_state.get("dli_entrypoint", "")
        entrypoint_args = st.session_state.get("dli_entrypoint_args", "")
        dependency_jars_raw = st.session_state.get("dli_dependency_jars", "")
        dependency_files_raw = st.session_state.get("dli_dependency_files", "")
        resume_checkpoint = st.session_state.get("dli_resume_checkpoint", False)
        checkpoint_path = st.session_state.get("dli_checkpoint_path", "")
        resume_max_num_input = st.session_state.get("dli_resume_max_num_text", "")

        if job_type == "sql":
            col_sql_1, col_sql_2 = st.columns(2)
            with col_sql_1:
                run_mode = st.selectbox(
                    "运行模式",
                    ["shared_cluster", "exclusive_cluster", "edge_node"],
                    index={
                        "shared_cluster": 0,
                        "exclusive_cluster": 1,
                        "edge_node": 2,
                    }.get(st.session_state.get("dli_run_mode", "shared_cluster"), 0),
                )
                st.caption(RUN_MODE_DESCRIPTIONS.get(run_mode, ""))
            with col_sql_2:
                checkpoint_enabled = st.checkbox(
                    "启用 Checkpoint",
                    value=st.session_state.get("dli_checkpoint_enabled", True),
                )
                checkpoint_interval = st.number_input(
                    "Checkpoint 间隔 (毫秒)",
                    min_value=1000,
                    step=1000,
                    value=st.session_state.get("dli_checkpoint_interval", 60000),
                )

            sql_body = st.text_area(
                "Flink SQL",
                value=st.session_state.get(
                    "dli_sql_body",
                    "-- 在此处输入 Flink SQL 语句\nSELECT 1;",
                ),
                height=260,
            )
        else:
            st.info(
                "Jar 作业会使用独享队列资源，请确保队列已绑定作业委托且资源包已上传至 DLI 资源目录。"
            )
            col_jar_1, col_jar_2 = st.columns(2)
            with col_jar_1:
                manager_cu_number = st.number_input(
                    "JobManager CU 数量",
                    min_value=1,
                    step=1,
                    value=st.session_state.get("dli_manager_cu_number", 1),
                )
                tm_cus_input = st.text_input(
                    "TaskManager CU 数量 (可选)",
                    value=st.session_state.get("dli_tm_cus_text", ""),
                    help="留空使用默认值；填写整数。",
                )
                tm_slot_input = st.text_input(
                    "TaskManager Slot 数 (可选)",
                    value=st.session_state.get("dli_tm_slot_text", ""),
                    help="留空使用默认值；填写整数。",
                )
                restart_when_exception = st.checkbox(
                    "发生异常自动重启",
                    value=st.session_state.get("dli_restart_when_exception", True),
                )
                resume_checkpoint = st.checkbox(
                    "自动从 Checkpoint 恢复",
                    value=st.session_state.get("dli_resume_checkpoint", False),
                )
            with col_jar_2:
                main_class = st.text_input(
                    "主类 (Main Class)",
                    value=st.session_state.get("dli_main_class", ""),
                    help="例如：com.example.JobMain",
                )
                entrypoint = st.text_input(
                    "入口 Jar (entrypoint)",
                    value=st.session_state.get("dli_entrypoint", ""),
                    help="填写 DLI 资源包路径，如 myGroup/job.jar。",
                )
                entrypoint_args = st.text_input(
                    "入口参数 (可选)",
                    value=st.session_state.get("dli_entrypoint_args", ""),
                    help="例如：--env prod --parallelism 2",
                )
                checkpoint_path = st.text_input(
                    "Checkpoint OBS 路径 (可选)",
                    value=st.session_state.get("dli_checkpoint_path", ""),
                    help="如启用快照或恢复，请填写 OBS 路径，例如 obs://bucket/path/",
                )
                resume_max_num_input = st.text_input(
                    "最大自动恢复次数 (可选)",
                    value=st.session_state.get("dli_resume_max_num_text", ""),
                )

            dependency_jars_raw = st.text_area(
                "依赖 Jar 列表（可选）",
                value=st.session_state.get("dli_dependency_jars", ""),
                height=120,
                help="每行一个资源路径，或用逗号分隔，例如 myGroup/lib1.jar。",
            )
            dependency_files_raw = st.text_area(
                "依赖文件列表（可选）",
                value=st.session_state.get("dli_dependency_files", ""),
                height=120,
                help="支持 CSV、配置文件等，每行一个资源路径。",
            )

        submitted = st.form_submit_button(
            "提交 Flink 作业",
            type="primary",
        )

    if submitted:
        st.session_state["dli_job_variant"] = job_type
        st.session_state["dli_job_name"] = job_name
        st.session_state["dli_cu_number"] = int(cu_number)
        st.session_state["dli_parallel_number"] = int(parallel_number)
        st.session_state["dli_log_enabled"] = log_enabled
        st.session_state["dli_auto_run"] = auto_run
        st.session_state["dli_runtime_config"] = runtime_config
        st.session_state["dli_flink_version"] = flink_version
        st.session_state["dli_execution_agency"] = execution_agency_urn
        if obs_bucket is not None:
            st.session_state["dli_obs_bucket"] = obs_bucket

        if job_type == "sql":
            st.session_state["dli_run_mode"] = run_mode
            st.session_state["dli_checkpoint_enabled"] = checkpoint_enabled
            st.session_state["dli_checkpoint_interval"] = int(checkpoint_interval)
            st.session_state["dli_sql_body"] = sql_body
        else:
            st.session_state["dli_manager_cu_number"] = manager_cu_number
            st.session_state["dli_tm_cus_text"] = tm_cus_input
            st.session_state["dli_tm_slot_text"] = tm_slot_input
            st.session_state["dli_restart_when_exception"] = restart_when_exception
            st.session_state["dli_main_class"] = main_class
            st.session_state["dli_entrypoint"] = entrypoint
            st.session_state["dli_entrypoint_args"] = entrypoint_args
            st.session_state["dli_dependency_jars"] = dependency_jars_raw
            st.session_state["dli_dependency_files"] = dependency_files_raw
            st.session_state["dli_resume_checkpoint"] = resume_checkpoint
            st.session_state["dli_checkpoint_path"] = checkpoint_path
            st.session_state["dli_resume_max_num_text"] = resume_max_num_input

        if log_enabled and not obs_bucket:
            st.error("启用作业日志时必须填写 OBS 桶名称。")
            return

        if job_type == "sql":
            if not sql_body or not sql_body.strip():
                st.error("请填写有效的 Flink SQL 语句。")
                return
            try:
                handle = client.submit_flink_sql_job(
                    name=job_name,
                    queue_name=queue_name,
                    sql_body=sql_body,
                    run_mode=run_mode,
                    cu_number=int(cu_number),
                    parallel_number=int(parallel_number) if parallel_number else None,
                    checkpoint_enabled=checkpoint_enabled,
                    checkpoint_interval=int(checkpoint_interval) if checkpoint_enabled else None,
                    log_enabled=log_enabled,
                    obs_bucket=obs_bucket or None,
                    runtime_config=runtime_config or None,
                    flink_version=flink_version or None,
                    execution_agency_urn=execution_agency_urn or None,
                )
            except DLIClientError as error:
                st.error(f"提交作业失败：{error}")
                return
        else:
            tm_cus = _parse_optional_int(tm_cus_input)
            tm_slot_num = _parse_optional_int(tm_slot_input)
            resume_max_num = _parse_optional_int(resume_max_num_input)

            if tm_cus_input.strip() and tm_cus is None:
                st.error("TaskManager CU 数量必须是整数。")
                return
            if tm_slot_input.strip() and tm_slot_num is None:
                st.error("TaskManager Slot 数必须是整数。")
                return
            if resume_max_num_input.strip() and resume_max_num is None:
                st.error("最大自动恢复次数必须是整数。")
                return
            if not main_class.strip():
                st.error("请填写主类 Main Class。")
                return
            if not entrypoint.strip():
                st.error("请填写入口 Jar (entrypoint)。")
                return
            if resume_checkpoint and not checkpoint_path.strip():
                st.error("启用自动恢复时必须指定 Checkpoint OBS 路径。")
                return

            dependency_jars = _parse_resource_list(dependency_jars_raw)
            dependency_files = _parse_resource_list(dependency_files_raw)

            try:
                handle = client.submit_flink_jar_job(
                    name=job_name,
                    queue_name=queue_name,
                    cu_number=int(cu_number),
                    manager_cu_number=int(manager_cu_number),
                    parallel_number=int(parallel_number) if parallel_number else None,
                    log_enabled=log_enabled,
                    obs_bucket=obs_bucket or None,
                    main_class=main_class.strip(),
                    entrypoint=entrypoint.strip(),
                    entrypoint_args=entrypoint_args.strip() or None,
                    dependency_jars=dependency_jars,
                    dependency_files=dependency_files,
                    restart_when_exception=bool(restart_when_exception),
                    tm_cus=tm_cus,
                    tm_slot_num=tm_slot_num,
                    flink_version=flink_version or None,
                    runtime_config=runtime_config or None,
                    execution_agency_urn=execution_agency_urn or None,
                    resume_checkpoint=bool(resume_checkpoint),
                    resume_max_num=resume_max_num,
                    checkpoint_path=checkpoint_path.strip() or None,
                )
            except DLIClientError as error:
                st.error(f"提交作业失败：{error}")
                return

        st.session_state["dli_last_job_id"] = handle.job_id
        st.session_state["dli_last_job_name"] = job_name
        st.success(f"作业 {handle.job_id} 已创建。")

        if auto_run:
            try:
                client.run_flink_job(handle.job_id)
                st.success(f"作业 {handle.job_id} 已提交并启动。")
            except DLIClientError as error:
                st.error(f"启动作业失败：{error}")

    st.subheader("作业状态")
    job_id = st.text_input(
        "查询作业 ID",
        value=st.session_state.get("dli_last_job_id", ""),
        help="可输入刚刚提交的作业 ID 或历史作业 ID。",
    )

    col_status, col_controls = st.columns([3, 1])
    with col_status:
        if st.button("刷新状态", use_container_width=True):
            if job_id:
                try:
                    st.session_state["dli_last_job_detail"] = client.get_job_detail(job_id)
                    st.success("已更新作业详情。")
                except DLIClientError as error:
                    st.error(f"获取作业详情失败：{error}")
            else:
                st.info("请先填写作业 ID。")
    with col_controls:
        if st.button("停止作业", use_container_width=True):
            if job_id:
                try:
                    client.stop_flink_job(job_id)
                    st.success("停止请求已发送。")
                except DLIClientError as error:
                    st.error(f"停止作业失败：{error}")
            else:
                st.info("请先填写作业 ID。")

    job_detail = st.session_state.get("dli_last_job_detail")
    if job_detail:
        job_type_label = job_detail.get("job_type") or "unknown"
        st.write(
            f"作业 `{job_detail['job_id']}` 类型：{job_type_label}，状态：{job_detail.get('status')} / {job_detail.get('status_desc')}"
        )
        st.json(job_detail)

    st.subheader("最近作业列表")
    if st.button("加载最近 20 个作业", use_container_width=True):
        try:
            jobs = client.list_jobs(queue_name=queue_name, limit=20)
            st.session_state["dli_job_list"] = jobs
            if jobs:
                st.success("已获取最新作业列表。")
            else:
                st.info("当前无历史作业。")
        except DLIClientError as error:
            st.error(f"获取作业列表失败：{error}")

    jobs = st.session_state.get("dli_job_list")
    if jobs:
        st.dataframe(jobs, use_container_width=True)


if __name__ == "__main__":
    main()
