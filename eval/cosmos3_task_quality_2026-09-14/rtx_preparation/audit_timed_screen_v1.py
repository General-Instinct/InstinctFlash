"""Local CPU replay of the fixed timed SCREEN; finite observer, no native actions."""

import argparse
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import time
import traceback


HERE = Path(__file__).resolve().parent
SCREEN_SHA = "83daaaa21c212ba7412a6a39b84487ba822d083d48980a0761fb1b3918280561"
CONFIG_SHA = "29644cc4deebc3b07d6a0ed0a59ea71b60f7f9c9286bdb48fe94f28391036bc0"
LAUNCH_SHA = "54cac2efa24ca67a46f3e59cbca0cd906323b41a43b665198c2472d43788a043"
DEADLINE = "2026-09-15T03:15:43Z"
FAMILIES = ("edge", "nano")
ARMS = ("baseline", "candidate")
ROUTES = ("eager_native", "runtime_selected")
ANCHORS = ("initial_state_sha256", "scene_config_sha256", "asset_inventory_sha256",
           "initial_nonvisual_observation_sha256", "initial_image_schema_sha256",
           "simulator_fingerprint_sha256")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def write_once(path, value, *, raw=False):
    """Create only a new audit artifact; never replace a controller artifact."""
    payload = value if raw else json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    with Path(path).open("x") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


@contextmanager
def readonly_anchor(formal, allowed_path):
    previous = formal.durable

    def check_existing(path, value, *args, **kwargs):
        require(Path(path) == allowed_path and allowed_path.is_file()
                and not allowed_path.is_symlink() and read(allowed_path) == value,
                "read-only collector may only verify its existing exact golden anchor")

    formal.durable = check_existing
    try:
        yield
    finally:
        formal.durable = previous


def actual_spec(path, expected, *, server=False):
    actual = read(path)
    require(set(actual) == set(expected), "copied specification fields differ")
    require(type(actual["max_seconds"]) is int and 0 < actual["max_seconds"] <= expected["max_seconds"],
            "copied job wall limit exceeds frozen native bound")
    normalized = dict(actual, max_seconds=expected["max_seconds"])
    if server:
        argv, frozen = list(actual["command"]), expected["command"]
        require(argv.count("--max-seconds") == frozen.count("--max-seconds") == 1,
                "one exact server budget argument required")
        index = frozen.index("--max-seconds") + 1
        require(argv.index("--max-seconds") + 1 == index and len(argv) == len(frozen), "server budget argument moved")
        budget = int(argv[index])
        require(argv[index] == str(budget) and 0 < budget <= int(frozen[index])
                and actual["max_seconds"] == budget + expected["grace_seconds"],
                "server supervisor and outer deadline budgets differ")
        argv[index] = frozen[index]
        normalized["command"] = argv
    require(normalized == expected,
            "copied native command, source, environment or ownership differs")
    return actual


def count_pairs(pairs):
    counts = dict.fromkeys(("both_success", "baseline_only_success", "candidate_only_success", "both_failure"), 0)
    for baseline, candidate in pairs:
        require(type(baseline) is bool and type(candidate) is bool, "native success must be boolean")
        counts["both_success" if baseline and candidate else "baseline_only_success" if baseline
               else "candidate_only_success" if candidate else "both_failure"] += 1
    return counts


def statistics(records, family, plan, api, *, complete=False):
    expected = plan["selection"]["pair_ids"]
    by_arm = {arm: {row["pair_id"]: row for row in records if row["family"] == family and row["arm"] == arm}
              for arm in ARMS}
    require(all(set(rows) <= set(expected) for rows in by_arm.values()), "unplanned paired key")
    if complete:
        require(all(set(rows) == set(expected) for rows in by_arm.values()), "complete SCREEN needs both exact six keys")

    def group(keys):
        matched = [key for key in keys if all(key in by_arm[arm] for arm in ARMS)]
        pairs = [(by_arm["baseline"][key]["success"], by_arm["candidate"][key]["success"]) for key in matched]
        absolute = {arm: {"completed": sum(key in rows for key in keys),
                          "successes": sum(rows[key]["success"] for key in keys if key in rows)}
                    for arm, rows in by_arm.items()}
        matched_absolute = {arm: {"completed": len(matched),
                                 "successes": sum(by_arm[arm][key]["success"] for key in matched)} for arm in ARMS}
        return {"planned_pairs": len(keys), "matched_pairs": len(matched), "absolute_counts": absolute,
                "matched_absolute_counts": matched_absolute, "paired_counts": count_pairs(pairs),
                "matched_candidate_minus_baseline": sum(int(b) - int(a) for a, b in pairs) / len(pairs) if pairs else None}, pairs

    result, pairs = group(expected)
    result.update(family=family, display_label="SMOKE SCREEN", status="SCREEN_COMPLETE" if complete else "INCOMPLETE_SCREEN",
                  task_quality_validated=False, certificate=None, fixed_family_gate_evaluated=complete,
                  fixed_family_gate_passed=None, lower_confidence_bound=None, upper_confidence_bound=None,
                  confidence_scope="Fixed six-pair family only; Tango one-sided 95% bounds, jointly central 90%; no full formal certificate")
    result["per_task"] = {task: group([row["pair_id"] for row in plan["selection"]["episodes"] if row["task_id"] == task])[0]
                          for task in plan["selection"]["task_ids"]}
    if complete:
        low, high = api._tango_paired_score_bounds(pairs, z=api.ONE_SIDED_95_Z)
        result.update(lower_confidence_bound=low, upper_confidence_bound=high, fixed_family_gate_passed=low > -0.05,
                      baseline_success=sum(a for a, _ in pairs) / 6, candidate_success=sum(b for _, b in pairs) / 6)
    return result


def compare_report(independent, published, protocol_hash, plan_hash):
    require(published["status"] == "SCREEN_COMPLETE" and published["display_label"] == "SMOKE SCREEN"
            and published["task_quality_validated"] is False and published["certificate"] is None
            and published["protocol_sha256"] == protocol_hash and published["screen_plan_sha256"] == plan_hash,
            "published family scope or native protocol differs")
    for key in ("family", "planned_pairs", "matched_pairs", "paired_counts", "absolute_counts", "baseline_success",
                "candidate_success", "lower_confidence_bound", "upper_confidence_bound", "fixed_family_gate_evaluated", "fixed_family_gate_passed"):
        require(published[key] == independent[key], "published family statistic differs: " + key)
    require(published["completed_baseline"] == published["completed_candidate"] == 6, "published six-pair coverage differs")
    require(published["per_task"] == {task: value["absolute_counts"] for task, value in independent["per_task"].items()},
            "published per-task absolute counts differ")


def csv_text(reports):
    columns = ("display_label", "family", "task", "audit_status", "planned_pairs", "matched_pairs",
               "baseline_completed", "baseline_successes", "candidate_completed", "candidate_successes",
               "matched_baseline_successes", "matched_candidate_successes", "candidate_minus_baseline_pp",
               "both_success", "baseline_only_success", "candidate_only_success", "both_failure",
               "family_lower_one_sided95_pp", "family_upper_one_sided95_pp", "fixed_family_gate_passed", "task_quality_validated")
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    for report in reports:
        for task, data in [("ALL_SELECTED_TASKS", report), *report["per_task"].items()]:
            whole = task == "ALL_SELECTED_TASKS"
            delta = data["matched_candidate_minus_baseline"]
            writer.writerow(dict(display_label="SMOKE SCREEN", family=report["family"], task=task,
                audit_status=report["audit_status"], planned_pairs=data["planned_pairs"], matched_pairs=data["matched_pairs"],
                baseline_completed=data["absolute_counts"]["baseline"]["completed"], baseline_successes=data["absolute_counts"]["baseline"]["successes"],
                candidate_completed=data["absolute_counts"]["candidate"]["completed"], candidate_successes=data["absolute_counts"]["candidate"]["successes"],
                matched_baseline_successes=data["matched_absolute_counts"]["baseline"]["successes"],
                matched_candidate_successes=data["matched_absolute_counts"]["candidate"]["successes"],
                candidate_minus_baseline_pp=100 * delta if delta is not None else "",
                family_lower_one_sided95_pp=100 * report["lower_confidence_bound"] if whole and report["lower_confidence_bound"] is not None else "",
                family_upper_one_sided95_pp=100 * report["upper_confidence_bound"] if whole and report["upper_confidence_bound"] is not None else "",
                fixed_family_gate_passed=report["fixed_family_gate_passed"] if whole and report["fixed_family_gate_evaluated"] else "",
                task_quality_validated=False, **data["paired_counts"]))
    return stream.getvalue()


class Audit:
    def __init__(self, config_path):
        require(sha(config_path) == CONFIG_SHA, "actual launched SCREEN config hash differs")
        require(sha(HERE / "run_timed_screen_v1.py") == SCREEN_SHA, "frozen SCREEN source differs")
        require(sha(HERE / "timed_screen_launch_v1.json") == LAUNCH_SHA, "actual launch receipt differs")
        spec = importlib.util.spec_from_file_location("readonly_screen_replay", HERE / "run_timed_screen_v1.py")
        screen = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = screen
        spec.loader.exec_module(screen)
        self.screen = screen
        self.config = read(config_path)
        values = screen.load_inputs(self.config)
        self.derived, self.plan, self.protocol, self.identities, self.helper, self.auditor, archive, self.api = values
        self.root = Path(self.config["roots"]["local"])
        launch = read(HERE / "timed_screen_launch_v1.json")
        self.launch = launch
        require(launch["output"] == str(self.root) and launch["source_sha256"] == SCREEN_SHA
                and launch["config_sha256"] == CONFIG_SHA and launch["plan_sha256"] == screen.PLAN_SHA,
                "launched root/source binding differs")
        for name, value in (("config.json", self.config), ("derived_execution_config.json", self.derived),
                            ("screen_plan.json", self.plan), ("protocol.json", self.protocol), ("identities.json", self.identities)):
            require(read(self.root / name) == value, "immutable startup copy differs: " + name)
        self.sources = {"orchestrator_sha256": SCREEN_SHA, "frozen_collector_sha256": screen.FORMAL_SHA,
                        "wrapper_sha256": screen.formal.WRAPPER_SHA, "transport_auditor_sha256": screen.formal.AUDITOR_SHA,
                        "config_sha256": CONFIG_SHA, "screen_plan_sha256": screen.PLAN_SHA}
        require(read(self.root / "source.json") == self.sources, "startup source binding differs")
        self.collector = screen.formal.FormalController(self.derived, self.protocol, self.identities, self.helper,
                                                       self.auditor, archive, self.api, None, self.root)

    def terminal(self):
        path = self.root / "completion.json"
        if not path.exists():
            return None
        try:
            return read(path)
        except json.JSONDecodeError:
            # Only the exact currently live producer's newest exclusive-write tail may be incomplete.
            try:
                stat = Path("/proc") / str(self.launch["pid"]) / "stat"
                fields = stat.read_text().rsplit(")", 1)[1].split()
                live = fields[0] not in {"Z", "X"} and fields[19] == str(self.launch["proc_start_ticks"])
            except FileNotFoundError:
                live = False
            require(live, "dead producer left malformed terminal evidence")
            return None

    def records(self):
        rows = read(self.root / "records.json") if (self.root / "records.json").exists() else []
        expected = [(family, arm, episode["pair_id"]) for family in FAMILIES for arm in ARMS for episode in self.plan["selection"]["episodes"]]
        require([(row["family"], row["arm"], row["pair_id"]) for row in rows] == expected[:len(rows)] and len(rows) <= 24,
                "SCREEN records duplicate, reorder, skip or add prospective keys")
        indexed = {row["pair_id"]: row for row in self.protocol["episodes"]}
        for row in rows:
            self.api._validate_record(row, self.protocol, indexed, self.api.digest(self.protocol))
        require(len({row["simulator_fingerprint_sha256"] for row in rows}) <= 1, "simulator source fingerprint differs between records")
        for pair in self.plan["selection"]["pair_ids"]:
            group = [row for row in rows if row["pair_id"] == pair]
            require(all(len({row[field] for row in group}) <= 1 for field in ANCHORS), "paired physical/source anchor differs")
        return rows

    def ready(self, family):
        path = self.root / (family + "_screen.json")
        if not path.exists() or read(path).get("status") != "SCREEN_COMPLETE":
            return False
        for route in ROUTES:
            cell = family + "-" + route
            directory = self.root / "batches/0000/servers" / cell
            if not (directory / "screen_close.json").exists():
                return False
            if read(directory / "screen_close.json").get("status") != "passed_complete_six":
                return False
            if not (directory / "final_server/completion.json").exists() or not (directory / "final_server/supervisor" / cell / "requests/closed.json").exists():
                return False
        return True

    def family(self, family, rows):
        require(self.ready(family), "family not closed and complete")
        selected = [row for row in rows if row["family"] == family]
        report = statistics(rows, family, self.plan, self.api, complete=True)
        episodes, closures = [], []
        for arm, route in zip(ARMS, ROUTES):
            cell = family + "-" + route
            server_expected = self.screen.specifications(self.derived, self.helper, cell)
            server_dir = self.root / server_expected["job_id"]
            final = server_dir / "final_server"
            server = actual_spec(final / "spec.json", server_expected, server=True)
            self.collector.validate_server_evidence(final, server, cell, closed=True)
            completion = read(final / "completion.json")
            require(completion["status"] in {"passed", "stopped"} and type(completion["exit_code"]) is int
                    and completion["exit_code"] == 0 and not any(completion.get(key) for key in
                        ("forced_kill", "cleanup_forced_kill", "owned_processes_still_live")), "unclean external server closure")
            cell_rows = [row for row in selected if row["arm"] == arm]
            requests = sum(row["generated_chunks"] for row in cell_rows)
            close = read(server_dir / "screen_close.json")
            require(close == {"cell": cell, "status": "passed_complete_six", "completed_records": 6,
                "expected_records": 6, "task_quality_validated": False, "archive_admission": False,
                "external_completion_sha256": sha(final / "completion.json"), "requests": requests, "episodes": 6},
                "SCREEN closure proof differs")
            ledger = final / "supervisor" / cell / "requests"
            require(read(ledger / "closed.json") == {"failed": False, "requests": requests, "episodes": 6,
                    "benchmark_identity_sha256": self.identities[cell]["benchmark_identity_sha256"]}, "closed ledger is not exactly six native streams")
            self.auditor.validate_ledger_files(ledger, requests, 6, closed=True)
            offset = 0
            for ordinal, (episode, persisted) in enumerate(zip(self.plan["selection"]["episodes"], cell_rows)):
                require(time.time() < deadline_unix(), "finite observer CPU deadline reached")
                job_expected = self.screen.specifications(self.derived, self.helper, cell, episode)
                directory = self.root / job_expected["job_id"]
                renderer = directory / "renderer"
                job = actual_spec(renderer / "spec.json", job_expected)
                self.collector.validate_owned_kit_log(renderer, job)
                self.auditor.validate_job_identity(renderer, job, read(directory / "launch.json"))
                self.collector.validate_server_evidence(directory / "thor_snapshot", server, cell)
                with readonly_anchor(self.screen.formal, self.root / "anchors" / (episode["pair_id"] + ".json")):
                    native = self.collector.collect_record(directory, read(renderer / "completion.json"), cell, episode)
                require(native == persisted == read(directory / "collected_record.json"), "persisted record differs from actual native outcome and external proof")
                instruction = next(row["instruction_default"] for row in self.protocol["inventory"]["tasks"] if row["task_id"] == episode["task_id"])
                replay = self.auditor.audit_transport(directory, native, episode, self.identities[cell], offset, ordinal, instruction)
                local = read(directory / "local_episode_audit.json")
                require(local == dict(replay, status="passed", task_quality_certified=False, orchestrator_source_sha256=SCREEN_SHA,
                                     collector_source_sha256=self.screen.FORMAL_SHA, transport_auditor_sha256=self.screen.formal.AUDITOR_SHA),
                        "local transport replay receipt differs")
                snapshot = directory / "thor_snapshot/supervisor" / cell / "requests"
                require(all(sha(path) == sha(ledger / path.name) for path in snapshot.iterdir()), "request source/action ledger changed before closure")
                episodes.append({"cell": cell, "pair_id": episode["pair_id"], "success": native["success"],
                    "executed_steps": native["executed_steps"], "generated_chunks": native["generated_chunks"],
                    "native_outcome": native["native_outcome"], "collected_record_sha256": sha(directory / "collected_record.json"),
                    "native_result_sha256": sha(renderer / "episode/result.json"), "transport_replay": replay})
                offset += native["generated_chunks"]
            closures.append({"cell": cell, "episodes": 6, "requests": requests,
                             "screen_close_sha256": sha(server_dir / "screen_close.json"),
                             "closed_ledger_sha256": sha(ledger / "closed.json"),
                             "external_completion_sha256": sha(final / "completion.json"), "single_native_worker_lifetime": True})
        published = read(self.root / (family + "_screen.json"))
        compare_report(report, published, self.api.digest(self.protocol), self.screen.PLAN_SHA)
        report.update(audit_status="PASSED_NATIVE_CLOSED_REPLAY", native_records_audited=len(episodes))
        return {"schema_version": 1, "kind": "independent_timed_SCREEN_family_audit_v1", "created_at_utc": now(),
                "sources": self.sources, "auditor_sha256": sha(__file__), "report": report,
                "published_family_report": published, "closures": closures, "episodes": episodes,
                "scope": "Local CPU replay of frozen V4 native outcome/source/physical observations, V1 transport input/action/per-request seed evidence, exact closed six-stream ledgers and independent paired math. No native execution, SSH, signals, retries, archive admission, latency benchmark, or full formal certification."}


def deadline_unix():
    return datetime.fromisoformat(DEADLINE.replace("Z", "+00:00")).timestamp()


def observe(audit, output, *, once=False):
    output.mkdir(exist_ok=False)
    completed = {}
    rows, terminal = [], None

    def audit_ready(rows):
        for family in FAMILIES:
            if family not in completed and audit.ready(family) and time.time() < deadline_unix():
                # SCREEN_COMPLETE is published after atomic records.json; take the rows after readiness.
                receipt = audit.family(family, audit.records())
                write_once(output / (family + "_independent_audit.json"), receipt)
                write_once(output / (family + "_audited_screen_table.csv"), csv_text([receipt["report"]]), raw=True)
                completed[family] = receipt["report"]
                print(json.dumps({"family": family, "status": "PASSED_NATIVE_CLOSED_REPLAY", "at": now()}), flush=True)

    try:
        while True:
            rows = audit.records()
            audit_ready(rows)
            terminal = audit.terminal()
            if terminal is not None:
                # Publication can complete after the first readiness pass. This is one new-family pass, never a retry.
                rows = audit.records()
                audit_ready(rows)
            if once or terminal is not None or time.time() >= deadline_unix():
                break
            time.sleep(min(20, max(0, deadline_unix() - time.time())))
        rows = audit.records()
        reports = []
        for family in FAMILIES:
            if family in completed:
                report = completed[family]
                require(statistics(rows, family, audit.plan, audit.api, complete=True) ==
                        {key: value for key, value in report.items() if key not in {"audit_status", "native_records_audited"}},
                        "completed family records changed after audit")
            else:
                report = statistics(rows, family, audit.plan, audit.api)
                report["audit_status"] = "NOT_AUDITED_COMPLETE_NATIVE_CLOSURE_UNAVAILABLE"
            reports.append(report)
        complete = len(rows) == 24 and len(completed) == 2 and terminal is not None and terminal.get("status") == "completed_fixed_SCREEN"
        summary = {"schema_version": 1, "kind": "independent_timed_SCREEN_final_audit_v1", "created_at_utc": now(),
                   "status": "PASSED_ALL_24_CLOSED_SCREEN" if complete else "PARTIAL_OR_NOT_FINISHED_SCREEN",
                   "display_label": "SMOKE SCREEN", "task_quality_validated": False, "certificate": None,
                   "controller_completion": terminal, "observer_deadline_utc": DEADLINE, "observed_records": len(rows),
                   "native_audited_families": list(completed), "reports": reports, "sources": audit.sources,
                   "auditor_sha256": sha(__file__), "raw_evidence_preserved": True,
                   "partial_counts_scope": "Persisted valid records only; incomplete families have no completed-family CI/gate and no independent closed native replay claim."}
        write_once(output / "independent_final_audit.json", summary)
        write_once(output / "audited_screen_table.csv", csv_text(reports), raw=True)
        return summary
    except BaseException:
        failure = {"status": "INDEPENDENT_AUDIT_FAILED", "created_at_utc": now(),
            "error": traceback.format_exc(), "sources": audit.sources, "auditor_sha256": sha(__file__),
            "completed_audited_families": list(completed), "task_quality_validated": False, "certificate": None,
            "raw_evidence_preserved": True, "automatic_retry": False}
        write_once(output / "audit_failure.json", failure)
        reports = []
        for family in FAMILIES:
            report = completed.get(family) or statistics(rows, family, audit.plan, audit.api)
            if family not in completed:
                report["audit_status"] = "FAILED_OR_UNAUDITED_NATIVE_REPLAY"
            reports.append(report)
        if not (output / "independent_final_audit.json").exists():
            write_once(output / "independent_final_audit.json", dict(failure, reports=reports))
        if not (output / "audited_screen_table.csv").exists():
            write_once(output / "audited_screen_table.csv", csv_text(reports), raw=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--observe", action="store_true")
    mode.add_argument("--once", action="store_true")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(sha(__file__) == args.source_sha256, "explicit auditor source hash differs")
    audit = Audit(args.config)
    require(args.output == audit.root / "independent_audit_v1", "only the separate prospective audit directory is writable")
    observe(audit, args.output, once=args.once)


if __name__ == "__main__":
    main()
