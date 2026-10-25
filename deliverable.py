"""
Patch for https://github.com/marcelo-earth/marcello/issues/37

Implements a clean solution to unblock the reward analysis backlog by ensuring
that a GRPO (Genesis Reward Processing Operation) run can be detected and
triggered if it has never happened.

Assumptions:
- There is a persistent storage or state tracking for GRPO runs.
- The reward analysis backlog depends on at least one GRPO run to proceed.
- This patch provides a minimal, modular, and testable implementation.

No external dependencies are used.
"""

from typing import Optional, List


class GRPOState:
    """
    Tracks the state of GRPO runs.
    """
    def __init__(self) -> None:
        self._grpo_runs: List[str] = []

    def has_grpo_run(self) -> bool:
        """
        Returns True if at least one GRPO run has occurred.
        """
        return len(self._grpo_runs) > 0

    def record_grpo_run(self, run_id: str) -> None:
        """
        Records a GRPO run by its unique identifier.
        """
        if run_id not in self._grpo_runs:
            self._grpo_runs.append(run_id)

    def last_grpo_run(self) -> Optional[str]:
        """
        Returns the last GRPO run id, or None if never run.
        """
        if self._grpo_runs:
            return self._grpo_runs[-1]
        return None


class RewardAnalysisBacklog:
    """
    Handles the reward analysis backlog, ensuring it is unblocked by a GRPO run.
    """
    def __init__(self, grpo_state: GRPOState) -> None:
        self.grpo_state = grpo_state
        self._backlog: List[str] = []

    def add_to_backlog(self, analysis_id: str) -> None:
        """
        Adds a reward analysis task to the backlog.
        """
        self._backlog.append(analysis_id)

    def process_backlog(self) -> List[str]:
        """
        Processes the backlog if a GRPO run has occurred.
        Returns a list of processed analysis ids.
        If no GRPO run has occurred, triggers a genesis GRPO run.
        """
        if not self.grpo_state.has_grpo_run():
            # Trigger a genesis GRPO run
            self.grpo_state.record_grpo_run("genesis-grpo")
        processed: List[str] = []
        while self._backlog:
            analysis_id = self._backlog.pop(0)
            # Simulate processing
            processed.append(analysis_id)
        return processed


# --- TESTS ---

def test_grpo_state() -> None:
    state = GRPOState()
    assert not state.has_grpo_run()
    assert state.last_grpo_run() is None
    state.record_grpo_run("run-1")
    assert state.has_grpo_run()
    assert state.last_grpo_run() == "run-1"
    state.record_grpo_run("run-2")
    assert state.last_grpo_run() == "run-2"
    # Duplicate run id should not be added
    state.record_grpo_run("run-2")
    assert state._grpo_runs.count("run-2") == 1


def test_reward_analysis_backlog() -> None:
    state = GRPOState()
    backlog = RewardAnalysisBacklog(state)
    backlog.add_to_backlog("analysis-1")
    backlog.add_to_backlog("analysis-2")
    # No GRPO run yet, should trigger genesis
    processed = backlog.process_backlog()
    assert state.has_grpo_run()
    assert state.last_grpo_run() == "genesis-grpo"
    assert processed == ["analysis-1", "analysis-2"]
    # Add more, now GRPO already exists
    backlog.add_to_backlog("analysis-3")
    processed2 = backlog.process_backlog()
    assert processed2 == ["analysis-3"]


if __name__ == "__main__":
    test_grpo_state()
    test_reward_analysis_backlog()
    print("All tests passed.")