import time
from dataclasses import dataclass, field
from typing import List
from datetime import datetime

COST_PER_1K_PROMPT_TOKENS = 0.003
COST_PER_1K_COMPLETION_TOKENS = 0.015
COST_PER_1K_CACHE_READ_TOKENS = 0.0003


@dataclass
class NodeMetric:
    node_name: str
    duration_seconds: float
    prompt_tokens: int
    completion_tokens: int
    cache_read_tokens: int = 0
    timestamp: str = field(
        default_factory=lambda: datetime.now().isoformat()
    )

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def cost_usd(self) -> float:
        prompt_cost = (self.prompt_tokens / 1000) * COST_PER_1K_PROMPT_TOKENS
        completion_cost = (
            self.completion_tokens / 1000
        ) * COST_PER_1K_COMPLETION_TOKENS
        cache_cost = (
            self.cache_read_tokens / 1000
        ) * COST_PER_1K_CACHE_READ_TOKENS
        return prompt_cost + completion_cost + cache_cost


class RunTracker:
    def __init__(self):
        self.metrics: List[NodeMetric] = []
        self.run_start: float = time.time()
        self.subtasks_completed: int = 0
        self.subtasks_failed: int = 0

    def record(
        self,
        node_name: str,
        duration: float,
        prompt_tokens: int,
        completion_tokens: int,
        cache_read_tokens: int = 0
    ):
        metric = NodeMetric(
            node_name=node_name,
            duration_seconds=duration,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cache_read_tokens=cache_read_tokens
        )
        self.metrics.append(metric)

    def mark_subtask_complete(self):
        self.subtasks_completed += 1

    def mark_subtask_failed(self):
        self.subtasks_failed += 1

    @property
    def total_duration(self) -> float:
        return time.time() - self.run_start

    @property
    def total_prompt_tokens(self) -> int:
        return sum(m.prompt_tokens for m in self.metrics)

    @property
    def total_completion_tokens(self) -> int:
        return sum(m.completion_tokens for m in self.metrics)

    @property
    def total_cache_tokens(self) -> int:
        return sum(m.cache_read_tokens for m in self.metrics)

    @property
    def total_tokens(self) -> int:
        return self.total_prompt_tokens + self.total_completion_tokens

    @property
    def total_cost(self) -> float:
        return sum(m.cost_usd for m in self.metrics)

    def node_summary(self) -> dict:
        nodes = {}
        for m in self.metrics:
            if m.node_name not in nodes:
                nodes[m.node_name] = {
                    "calls": 0,
                    "total_duration": 0.0,
                    "total_tokens": 0,
                }
            nodes[m.node_name]["calls"] += 1
            nodes[m.node_name]["total_duration"] += m.duration_seconds
            nodes[m.node_name]["total_tokens"] += m.total_tokens
        return nodes

    def print_summary(self):
        node_stats = self.node_summary()

        print("\n" + "=" * 60)
        print("RUN SUMMARY")
        print("=" * 60)
        print(f"  Total time:          {self.total_duration:.1f}s")
        print(f"  Total tokens:        {self.total_tokens:,}")
        print(f"    Prompt tokens:     {self.total_prompt_tokens:,}")
        print(f"    Completion tokens: {self.total_completion_tokens:,}")
        if self.total_cache_tokens:
            print(f"    Cache read tokens: {self.total_cache_tokens:,}")
        print(f"  Estimated cost:      ${self.total_cost:.4f}")
        print(f"  Model calls:         {len(self.metrics)}")
        print()
        print("  Per-node breakdown:")
        for node_name, stats in node_stats.items():
            calls = stats["calls"]
            avg_duration = stats["total_duration"] / calls
            avg_tokens = stats["total_tokens"] // calls
            token_str = (
                f"avg {avg_tokens:,} tokens"
                if avg_tokens > 0
                else "no model calls"
            )
            print(
                f"    {node_name:<16} "
                f"{calls:>2} calls  "
                f"avg {avg_duration:.1f}s  "
                f"{token_str}"
            )
        print()
        print(
            f"  Subtasks: {self.subtasks_completed} completed, "
            f"{self.subtasks_failed} failed"
        )
        print("=" * 60)