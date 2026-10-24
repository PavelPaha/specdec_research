"""An early completion must admit the next prompt before a straggler finishes."""

from types import SimpleNamespace

from benchmarks.long_context.continuous import drive


def test_refills_slot_before_waiting_for_straggler():
    class Engine:
        def __init__(self):
            self.active = {}
            self.history = []

        def has_unfinished_requests(self):
            return bool(self.active)

        def step(self):
            self.history.append(set(self.active))
            results = []
            for rid in list(self.active):
                self.active[rid] -= 1
                done = self.active[rid] == 0
                results.append(SimpleNamespace(id=rid, finished=done))
                if done:
                    del self.active[rid]
            return results

    engine = Engine()
    completed = []

    def submit(prompt):
        rid, length = prompt
        engine.active[rid] = length
        return rid

    def consume(result):
        if result.finished:
            completed.append(result.id)
            return result.id
        return None

    drive(engine, [("slow", 4), ("fast", 1), ("next", 2), ("last", 1)], 2, submit, consume)
    assert engine.history[0] == {"slow", "fast"}
    assert engine.history[1] == {"slow", "next"}
    assert max(map(len, engine.history)) == 2
    assert sorted(completed) == ["fast", "last", "next", "slow"]
