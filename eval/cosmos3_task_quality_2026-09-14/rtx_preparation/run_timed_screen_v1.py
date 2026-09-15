"""Fixed24-episode SCREEN with an absolute work/cleanup deadline; no retries."""

import argparse
import copy
from datetime import datetime
import hashlib
import importlib.util
import math
from pathlib import Path
import signal
import sys
import time
import traceback


HERE = Path(__file__).resolve().parent
FORMAL_SHA = "390fdad55769473d43ef2eb5eb538de81f04c27fd6bfaf7c8101b96c32876953"
ORIGINAL_SHA = "910c104756a10508559ab7f883d5e1f7b263b7decaff9908956d37b76ad5d1ae"
PLAN_SHA = "5a72212c2e3185ea97e391737dd632eb5f0d8d6a56bb605593e961afacfdc8e1"
DEADLINE_UTC = "2026-09-15T03:10:43Z"
CLEANUP_SECONDS = 300
if hashlib.sha256((HERE / "run_paired_formal_v4.py").read_bytes()).hexdigest() != FORMAL_SHA:
    raise ValueError("frozen collector source changed before import")
spec = importlib.util.spec_from_file_location("timed_screen_frozen_formal_v4", HERE / "run_paired_formal_v4.py")
formal = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = formal
spec.loader.exec_module(formal)
formal.require(formal.sha(HERE / "run_paired_formal_v4.py") == FORMAL_SHA, "frozen collector source changed")
read, sha, durable, require = formal.read, formal.sha, formal.durable, formal.require


class ScreenDeadline(TimeoutError):
    pass


def cutoff(config):
    return datetime.fromisoformat(config["deadline_utc"].replace("Z", "+00:00")).timestamp()


def load_inputs(config):
    require(set(config) == {"schema_version", "kind", "original_config", "screen_plan", "run_id", "roots",
                            "deadline_utc", "server_max_seconds"}, "explicit bounded SCREEN config fields required")
    require(config["schema_version"] == 1 and config["kind"] == "cosmos_robolab_timed_screen_v1"
            and config["deadline_utc"] == DEADLINE_UTC and config["server_max_seconds"] == 14400,
            "SCREEN scope or deadline differs")
    require(config["original_config"]["sha256"] == ORIGINAL_SHA and config["screen_plan"]["sha256"] == PLAN_SHA,
            "SCREEN must bind the exact original source/config and prospective plan")
    for reference in (config["original_config"], config["screen_plan"]):
        require(sha(reference["local"]) == reference["sha256"], "bound SCREEN input changed")
    original, plan = read(config["original_config"]["local"]), read(config["screen_plan"]["local"])
    protocol, identities, helper, auditor, archive, api = formal.load_inputs(original, ORIGINAL_SHA, for_execution=True)
    require(plan["kind"] == "cosmos_robolab_fixed_smoke_screen_plan_v1"
            and plan["selection"]["total_native_episodes"] == 24
            and plan["selection"]["fixed_pairs_per_family"] == 6
            and plan["selection"]["old_smoke_or_formal_outcomes_adopted"] == 0
            and plan["deadline"]["fixed_cutoff_utc"] == DEADLINE_UTC,
            "prospective SCREEN selection differs")
    selected = plan["selection"]["episodes"]
    indexed = {row["pair_id"]: row for row in protocol["episodes"]}
    require(len(selected) == 6 and len({row["pair_id"] for row in selected}) == 6
            and all(indexed.get(row["pair_id"]) == row for row in selected), "selected rows differ from original protocol")
    require(plan["selection"]["pair_ids"] == [row["pair_id"] for row in selected]
            and plan["execution_priority"]["cell_order"] == list(formal.CELLS), "fixed order differs")
    require(set(config["roots"]) == {"local", "thor", "renderer"} and isinstance(config["run_id"], str)
            and config["run_id"].strip() and config["run_id"] != original["run_id"], "fresh named roots required")
    for host, value in config["roots"].items():
        path = Path(value)
        old = Path(config["original_config"]["local"]).parent.parent / "paired_formal_v2" if host == "local" else Path(original[host]["output_root"])
        require(path.is_absolute() and ".." not in path.parts and not path.is_relative_to(old)
                and not old.is_relative_to(path) and not any(ch.isspace() for ch in value), "new root overlaps prior work")
    derived = copy.deepcopy(original)
    derived["run_id"] = config["run_id"]
    for host in ("thor", "renderer"):
        derived[host]["output_root"] = config["roots"][host]
    derived["limits"]["server_seconds"] = config["server_max_seconds"]
    return derived, plan, protocol, identities, helper, auditor, archive, api


def specifications(config, helper, cell, episode=None, *, remaining=None):
    """Frozen native argv/body; only new paths and explicit wall-clock budget differ."""
    temporary = copy.deepcopy(config)
    if remaining is not None:
        require(remaining > 0, "no native job may start after the work cutoff")
        key = "server_seconds" if episode is None else "episode_seconds"
        temporary["limits"][key] = max(1, min(temporary["limits"][key], math.floor(remaining)))
    return formal.specifications(temporary, helper, 0, cell, episode)


class TimedTransport(formal.FormalTransport):
    def __init__(self, config, output, helper, deadline):
        super().__init__(config, output, helper)
        self.deadline, self.cleanup = deadline, False

    def remaining(self):
        return self.deadline - (0 if self.cleanup else CLEANUP_SECONDS) - time.time()

    def invoke(self, *args, **kwargs):
        remaining = self.remaining()
        if remaining <= 0:
            raise ScreenDeadline("absolute SCREEN work/cleanup deadline reached")
        limits = self.config["limits"]
        prior = limits["control_seconds"]
        limits["control_seconds"] = min(prior, remaining)
        try:
            return super().invoke(*args, **kwargs)
        finally:
            limits["control_seconds"] = prior

    def copy(self, *args):
        remaining = self.remaining()
        if remaining <= 0:
            raise ScreenDeadline("absolute SCREEN copy deadline reached; remote evidence retained")
        limits = self.config["limits"]
        prior = limits["copy_seconds"]
        limits["copy_seconds"] = min(prior, remaining)
        try:
            return super().copy(*args)
        finally:
            limits["copy_seconds"] = prior


class ScreenController(formal.FormalController):
    def __init__(self, screen, config, plan, protocol, identities, helper, auditor, archive, api, transport, output):
        super().__init__(config, protocol, identities, helper, auditor, archive, api, transport, output)
        self.screen, self.plan = screen, plan
        self.episodes = plan["selection"]["episodes"]
        self.deadline = cutoff(screen)
        self.closed_cells = {}
        self.cleanup_attempted = set()
        self.cleanup_completions = {}
        self.state = {"status": "prepared_SCREEN", "expected_records": 24, "completed_records": 0,
                      "deadline_utc": DEADLINE_UTC, "cleanup_reserve_seconds": CLEANUP_SECONDS,
                      "automatic_retry": False, "automatic_follow_on": False, "task_quality_validated": False,
                      "original_formal_protocol_unchanged": True, "archive_admission": False, "reclamation": False}

    def remaining(self):
        return self.transport.remaining()

    def work_guard(self):
        if self.deadline - CLEANUP_SECONDS - time.time() <= 0:
            raise ScreenDeadline("stop native work at the fixed cutoff minus300seconds")

    def wait(self, host, spec, seconds):
        end = min(time.time() + seconds, time.time() + self.remaining())
        while True:
            state = self.transport.status(host, spec["output"])
            if state["completion"] is not None:
                return state["completion"]
            require(state["live"], "owned native job ended without completion")
            if time.time() >= end:
                raise ScreenDeadline("bounded native wait ended; no retry")
            time.sleep(min(self.config["limits"]["poll_seconds"], max(0, end - time.time())))

    def prepare_hosts(self):
        for host in ("thor", "renderer"):
            self.transport.prepare(host, self.config[host]["output_root"])
            durable(self.output / (host + "_root_prepared.json"), {"host": host, "root": self.config[host]["output_root"],
                "fresh_exclusive_directory": True, "archive_admission": False, "reclamation": False})

    def collect_screen_episode(self, cell, episode, ordinal, offset, server):
        self.work_guard()
        job = specifications(self.config, self.helper, cell, episode, remaining=self.remaining())
        directory = self.output / job["job_id"]
        formal.durable_directories(directory, exclusive=True)
        self.ensure_capacity("renderer", directory)
        self.renderer = job
        durable(directory / "launch.json", self.transport.spawn("renderer", job))
        complete = self.wait("renderer", job, self.config["limits"]["episode_seconds"] + self.config["limits"]["grace_seconds"])
        self.transport.copy("renderer", job["output"], directory / "renderer")
        self.renderer = None
        self.transport.copy("thor", server["output"], directory / "thor_snapshot")
        self.validate_owned_kit_log(directory / "renderer", job)
        anchor = self.transport.read_json("renderer", self.config["renderer"]["output_root"] + "/anchors/" + episode["pair_id"] + ".json")
        durable(directory / "native_anchor.json", anchor)
        self.auditor.validate_job_identity(directory / "renderer", job, read(directory / "launch.json"))
        self.validate_server_evidence(directory / "thor_snapshot", server, cell)
        row = self.collect_record(directory, complete, cell, episode)
        instruction = next(t["instruction_default"] for t in self.protocol["inventory"]["tasks"] if t["task_id"] == episode["task_id"])
        replay = self.auditor.audit_transport(directory, row, episode, self.identities[cell], offset, ordinal, instruction)
        durable(directory / "local_episode_audit.json", {**replay, "status": "passed", "task_quality_certified": False,
                "orchestrator_source_sha256": sha(__file__), "collector_source_sha256": FORMAL_SHA,
                "transport_auditor_sha256": formal.AUDITOR_SHA})
        durable(directory / "collected_record.json", row)
        self.records.append(row)
        durable(self.output / "records.json", self.records, exclusive=False)
        self.state["completed_records"] = len(self.records)
        self.save()
        self.screen_reports()
        return row

    def stop_owned_for_cleanup(self):
        """Attempt both owned stops independently; preserve any unclosed pointers."""
        self.transport.cleanup = True
        for host, attribute in (("renderer", "renderer"), ("thor", "server")):
            job = getattr(self, attribute)
            if job is None or (host, job["job_id"]) in self.cleanup_attempted:
                continue
            self.cleanup_attempted.add((host, job["job_id"]))
            try:
                completion = self.stop(host, job)
                durable(self.output / "owned_cleanup" / host / (job["job_id"] + ".json"),
                        {"completion": completion, "spec": job, "task_quality_validated": False})
                require(completion.get("exit_code") == 0 and not completion.get("owned_processes_still_live")
                        and not completion.get("forced_kill") and not completion.get("cleanup_forced_kill"),
                        "owned native process lacks a clean external exit")
                self.cleanup_completions[(host, job["job_id"])] = completion
                setattr(self, attribute, None)
            except BaseException:
                self.state.setdefault("cleanup_errors", []).append({"host": host, "job_id": job["job_id"],
                    "remote_output": job["output"], "error": traceback.format_exc()})

    def close_screen_cell(self, server, cell, rows):
        previous = self.transport.cleanup
        interrupted_renderer = self.renderer
        directory = self.output / server["job_id"]
        result = {"cell": cell, "status": "partial_or_failed", "completed_records": len(rows),
                  "expected_records": 6, "task_quality_validated": False, "archive_admission": False}
        try:
            self.stop_owned_for_cleanup()
            copy_errors = []
            if interrupted_renderer is not None:
                try:
                    self.transport.copy("renderer", interrupted_renderer["output"],
                                        self.output / "excluded_attempts" / interrupted_renderer["job_id"])
                except BaseException:
                    copy_errors.append({"host": "renderer", "remote_output": interrupted_renderer["output"],
                                        "error": traceback.format_exc()})
            final = directory / "final_server"
            try:
                self.transport.copy("thor", server["output"], final)
            except BaseException:
                copy_errors.append({"host": "thor", "remote_output": server["output"], "error": traceback.format_exc()})
            if copy_errors:
                self.state.setdefault("cleanup_copy_errors", []).extend(copy_errors)
                raise RuntimeError("owned native evidence copy failed; retained local partial and remote originals")
            complete = self.cleanup_completions.get(("thor", server["job_id"]))
            require(complete is not None, "server clean external closure was not proved")
            self.validate_server_evidence(final, server, cell, closed=True)
            require(read(final / "completion.json") == complete and complete["status"] in {"passed", "stopped"}
                    and complete.get("exit_code") == 0 and not complete.get("owned_processes_still_live")
                    and not complete.get("forced_kill") and not complete.get("cleanup_forced_kill"), "clean external server exit required")
            result["external_completion_sha256"] = sha(final / "completion.json")
            if len(rows) == 6:
                ledger = final / "supervisor" / cell / "requests"
                count = sum(row["generated_chunks"] for row in rows)
                require(read(ledger / "closed.json") == {"failed": False, "requests": count, "episodes": 6,
                        "benchmark_identity_sha256": self.identities[cell]["benchmark_identity_sha256"]}, "exact six-stream close required")
                self.auditor.validate_ledger_files(ledger, count, 6, closed=True)
                require([row["pair_id"] for row in rows] == self.plan["selection"]["pair_ids"], "six fixed paired keys required")
                offset = 0
                for ordinal, (episode, row) in enumerate(zip(self.episodes, rows)):
                    episode_dir = self.output / specifications(self.config, self.helper, cell, episode)["job_id"]
                    before = episode_dir / "thor_snapshot/supervisor" / cell / "requests"
                    require(all(sha(path) == sha(ledger / path.name) for path in before.iterdir()), "immutable per-request source/action ledger changed")
                    instruction = next(t["instruction_default"] for t in self.protocol["inventory"]["tasks"] if t["task_id"] == episode["task_id"])
                    replay = self.auditor.audit_transport(episode_dir, row, episode, self.identities[cell], offset, ordinal, instruction)
                    require(all(read(episode_dir / "local_episode_audit.json").get(k) == v for k, v in replay.items()), "closed transport replay differs")
                    offset += row["generated_chunks"]
                result.update(status="passed_complete_six", requests=count, episodes=6)
        except BaseException:
            result["error"] = traceback.format_exc()
            raise
        finally:
            self.closed_cells[cell] = result
            durable(directory / "screen_close.json", result)
            self.transport.cleanup = previous
            self.screen_reports()

    def screen_reports(self):
        expected = self.plan["selection"]["pair_ids"]
        fields = ("initial_state_sha256", "scene_config_sha256", "asset_inventory_sha256",
                  "initial_nonvisual_observation_sha256", "initial_image_schema_sha256", "simulator_fingerprint_sha256")
        require(len({row["simulator_fingerprint_sha256"] for row in self.records}) <= 1,
                "SCREEN records mix native simulator fingerprints")
        keys = [(row["family"], row["arm"], row["pair_id"]) for row in self.records]
        require(len(keys) == len(set(keys)), "duplicate SCREEN record")
        for family in ("edge", "nano"):
            for arm in ("baseline", "candidate"):
                group = [row for row in self.records if row["family"] == family and row["arm"] == arm]
                require(all(row["execution_binding_sha256"] == self.protocol["execution_bindings"][family][arm]
                            for row in group), "SCREEN arm does not match the exact frozen deployment")
        for pair in expected:
            group = [row for row in self.records if row["pair_id"] == pair]
            require(all(len({row[name] for row in group}) <= 1 for name in fields),
                    "available four-arm SCREEN physical/source anchors differ")
        reports = {}
        for family in ("edge", "nano"):
            by_arm = {arm: {row["pair_id"]: row for row in self.records if row["family"] == family and row["arm"] == arm}
                      for arm in ("baseline", "candidate")}
            matched = [pair for pair in expected if pair in by_arm["baseline"] and pair in by_arm["candidate"]]
            for pair in matched:
                require(all(by_arm["baseline"][pair][name] == by_arm["candidate"][pair][name] for name in fields), "paired SCREEN anchor/source mismatch")
            complete = (all(set(rows) == set(expected) for rows in by_arm.values())
                        and all(self.closed_cells.get(family + "-" + arm, {}).get("status") == "passed_complete_six"
                                for arm in ("eager_native", "runtime_selected")))
            counts = {"both_success": 0, "baseline_only_success": 0, "candidate_only_success": 0, "both_failure": 0}
            pairs = []
            for pair in matched:
                a, b = by_arm["baseline"][pair]["success"], by_arm["candidate"][pair]["success"]
                pairs.append((a, b))
                counts["both_success" if a and b else "baseline_only_success" if a else "candidate_only_success" if b else "both_failure"] += 1
            report = {"status": "SCREEN_COMPLETE" if complete else "PENDING_INCOMPLETE_COVERAGE", "display_label": "SMOKE SCREEN",
                      "family": family, "planned_pairs": 6, "completed_baseline": len(by_arm["baseline"]),
                      "completed_candidate": len(by_arm["candidate"]), "matched_pairs": len(matched), "paired_counts": counts,
                      "task_quality_validated": False, "certificate": None, "fixed_family_gate_evaluated": False,
                      "protocol_sha256": self.api.digest(self.protocol), "screen_plan_sha256": PLAN_SHA,
                      "limits": "Two tasks/three fixed fresh seeds; never a full formal certificate. Partial coverage cannot evaluate the fixed-N gate."}
            report["absolute_counts"] = {arm: {"completed": len(rows), "successes": sum(row["success"] for row in rows.values())}
                                         for arm, rows in by_arm.items()}
            report["per_task"] = {task: {arm: {"completed": sum(row["task_id"] == task for row in rows.values()),
                "successes": sum(row["success"] for row in rows.values() if row["task_id"] == task)}
                for arm, rows in by_arm.items()} for task in self.plan["selection"]["task_ids"]}
            if complete:
                low, high = self.api._tango_paired_score_bounds(pairs, z=self.api.ONE_SIDED_95_Z)
                report.update(baseline_success=sum(a for a, _ in pairs) / 6, candidate_success=sum(b for _, b in pairs) / 6,
                              lower_confidence_bound=low, upper_confidence_bound=high, interval="Tango one-sided95 bounds; jointly central90",
                              fixed_family_gate_evaluated=True, fixed_family_gate_passed=low > -0.05)
            reports[family] = report
            durable(self.output / (family + "_screen.json"), report, exclusive=False)
        durable(self.output / "screen_summary.json", {"status": self.state["status"], "records": len(self.records),
                "expected_records": 24, "families": reports, "closed_cells": self.closed_cells,
                "task_quality_validated": False, "deadline_utc": DEADLINE_UTC}, exclusive=False)

    def run(self):
        self.save()
        self.screen_reports()
        try:
            self.work_guard()
            self.prepare_hosts()
            for cell in formal.CELLS:
                self.work_guard()
                server = specifications(self.config, self.helper, cell, remaining=self.remaining())
                directory = self.output / server["job_id"]
                formal.durable_directories(directory, exclusive=True)
                self.ensure_capacity("thor", directory)
                self.server = server
                rows = []
                try:
                    durable(directory / "launch.json", self.transport.spawn("thor", server))
                    durable(directory / "ready.json", self.ready(server, cell))
                    for ordinal, episode in enumerate(self.episodes):
                        self.work_guard()
                        self.state.update(status="running_SCREEN_episode", cell=cell, pair_id=episode["pair_id"])
                        self.save()
                        try:
                            rows.append(self.collect_screen_episode(cell, episode, ordinal, sum(row["generated_chunks"] for row in rows), server))
                        except BaseException:
                            durable(self.output / "excluded_attempts" / (cell + "-" + str(ordinal) + ".json"),
                                    {"pair_id": episode["pair_id"], "cell": cell, "excluded_from_success_counts": True,
                                     "error": traceback.format_exc(), "automatic_retry": False})
                            raise
                finally:
                    # Publish partial results before spending the reserved cleanup interval.
                    try:
                        self.screen_reports()
                    finally:
                        self.close_screen_cell(server, cell, rows)
            require(len(self.records) == 24 and all(value["status"] == "passed_complete_six" for value in self.closed_cells.values()), "complete24 requires all4 native closures")
            self.state["status"] = "completed_fixed_SCREEN"
        except BaseException:
            self.state.update(status="stopped_partial_SCREEN", error=traceback.format_exc())
            raise
        finally:
            self.stop_owned_for_cleanup()
            self.state["unclosed_owned_specs"] = {host: job for host, job in (("renderer", self.renderer), ("thor", self.server)) if job is not None}
            self.state["ended_unix"] = time.time()
            self.state["remote_evidence_retained"] = True
            self.save()
            try:
                self.screen_reports()
            finally:
                durable(self.output / "completion.json", self.state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(sha(args.config) == args.config_sha256, "explicit SCREEN configuration hash required")
    screen = read(args.config)
    values = load_inputs(screen)
    config, plan, protocol, identities, helper, auditor, archive, api = values
    require(not args.execute or args.output == Path(screen["roots"]["local"]), "exact local run root required")
    formal.durable_directories(args.output, exclusive=True)
    for name, value in (("config.json", screen), ("derived_execution_config.json", config), ("screen_plan.json", plan),
                        ("protocol.json", protocol), ("identities.json", identities)):
        durable(args.output / name, value)
    durable(args.output / "source.json", {"orchestrator_sha256": sha(__file__), "frozen_collector_sha256": FORMAL_SHA,
            "wrapper_sha256": formal.WRAPPER_SHA, "transport_auditor_sha256": formal.AUDITOR_SHA,
            "config_sha256": args.config_sha256, "screen_plan_sha256": PLAN_SHA})
    if args.prepare:
        jobs = [specifications(config, helper, cell, episode) for cell in formal.CELLS for episode in plan["selection"]["episodes"]]
        require(len(jobs) == len({job["job_id"] for job in jobs}) == 24, "exact24 unique prospective jobs required")
        durable(args.output / "prepared.json", {"status": "CPU_prepared_only", "episodes": 24, "servers": 4,
                "job_ids": [job["job_id"] for job in jobs], "deadline_utc": DEADLINE_UTC, "native_launches": 0})
        return

    def interrupted(number, _frame):
        raise InterruptedError(f"SCREEN controller received {signal.Signals(number).name}")
    for number in (signal.SIGTERM, signal.SIGINT):
        signal.signal(number, interrupted)
    transport = TimedTransport(config, args.output, helper, cutoff(screen))
    ScreenController(screen, config, plan, protocol, identities, helper, auditor, archive, api, transport, args.output).run()


if __name__ == "__main__":
    main()
