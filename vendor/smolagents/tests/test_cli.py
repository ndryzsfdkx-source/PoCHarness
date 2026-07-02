from smolagents.cli import _collect_artifacts, _create_task_prompt


def test_collect_artifacts_uses_configured_runtime_log_dirs(tmp_path):
    class FakeExecResult:
        exit_code = 0
        output = b""

    class FakeContainer:
        def exec_run(self, *args, **kwargs):
            return FakeExecResult()

    class FakeRuntime:
        container = FakeContainer()
        workdir = "/src/app"

        def __init__(self):
            self.commands = []

        def exec(self, command):
            self.commands.append(command)
            return FakeExecResult()

    runtime = FakeRuntime()
    instance_dir = tmp_path / "instance"
    (instance_dir / "artifacts").mkdir(parents=True)

    _collect_artifacts(
        {"work_dir": "/src/app"},
        instance_dir,
        "poc-san",
        runtime,
        config={
            "agent": {
                "pocharness": {"log_dir": "/tmp/synthesis_top3"},
            }
        },
    )

    shell_commands = [command[2] for command in runtime.commands if command[:2] == ["bash", "-c"]]
    joined = "\n".join(shell_commands)

    assert "cp /tmp/synthesis_top3/synthesis_log.jsonl" in joined
    assert "synthesis_attempt_ledger.jsonl" not in joined
    assert "cp /tmp/synthesis_top3/synthesis_finalization_guard.jsonl" in joined
    assert "cp /tmp/synthesis_top3/finalization_review.jsonl" in joined


def test_poc_desc_floor_prompt_includes_unchanged_base_prompt():
    instance = {
        "work_dir": "/src/example",
        "bug_description": "confirmed bug",
    }
    base = _create_task_prompt(instance, {"type": "poc-desc"}, max_steps=75)
    extended = _create_task_prompt(
        instance,
        {
            "type": "poc-desc",
            "prompt_template": "poc-desc-floor.j2",
        },
        max_steps=75,
    )

    assert extended.startswith(base.rstrip())
    assert "Additional evaluation constraint - independent synthesis" in extended
    assert "its existing local history" in extended
    assert "attachments, reference PoCs, or fetch additional Git history" in extended


def test_poc_desc_abc_prompt_wires_synthesis_helper_and_hides_sanitizer_report():
    prompt = _create_task_prompt(
        {
            "work_dir": "/src/example",
            "bug_description": "heap overflow in target_function",
            "sanitizer_report": "must remain hidden",
        },
        {
            "type": "poc-desc",
            "prompt_template": "poc-desc-pocharness.j2",
        },
        max_steps=75,
        config={
            "agent": {
                "pocharness": {
                    "enabled": True,
                    "reviewer": {"enabled": True},
                }
            }
        },
    )

    assert "Call `artifact_guard`" in prompt
    assert "Call `final_submission" in prompt
    assert "request_synthesis_helper" in prompt
    assert "delegated_problem" in prompt
    # poc-desc description_mode never exposes the dataset sanitizer_report to A.
    assert "must remain hidden" not in prompt
