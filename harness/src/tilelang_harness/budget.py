"""Write-ahead counters for framework-controlled vendor builds and kernel calls."""

from .artifacts import write_json
from .contracts import ContractError, integer


class BudgetExhausted(ContractError):
    pass


def limits_for(job):
    return job.get(
        "resource_limits",
        {
            "launches": 3 if job["mode"] == "device-smoke" else 0,
            "vendor_builds": (2 if job["measure"] else 1) if job["mode"] == "device-smoke" else 0,
        },
    )


class ResourceBudget:
    def __init__(self, limits, path=None):
        if set(limits) != {"launches", "vendor_builds"}:
            raise ContractError("invalid resource limits")
        self.limits = {key: integer(value, key, 0) for key, value in limits.items()}
        self.used = dict.fromkeys(limits, 0)
        self.path = path
        self.save()

    def consume(self, kind):
        if self.used[kind] >= self.limits[kind]:
            raise BudgetExhausted(f"{kind} budget exhausted")
        self.used[kind] += 1
        self.save()  # Count before calling vendor code, including failed/ambiguous calls.

    def snapshot(self):
        return {"limits": dict(self.limits), "used": dict(self.used), "accounting_complete": True}

    def save(self):
        if self.path is not None:
            write_json(self.path, self.snapshot())
