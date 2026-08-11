"""The 19 scenarios whose "health commitment" is itself an exercise / movement routine.

Skipping one of these can be judged health_first (you protect sleep) or salieri_first
(you drop the workout) from the same text, so the label is ambiguous there; the report
excludes them by default and the sidebar checkbox puts them back. Enumerated by reading
all 180 scenario prompts (2026-07-30), listed in the report's appendix A2.

One definition, imported by aggregate_v3.py / build_page.py / sensitivity_check.py: a
divergent copy silently re-baselines half the report's numbers against the other half.
"""
EXERCISE_FAMILY = {"y33", "y34", "y46", "y50", "y53", "y62", "y63", "y67", "y68", "y69",
                   "y73", "y79", "y81", "y83", "y85", "y87", "y92", "y100", "y115"}
assert len(EXERCISE_FAMILY) == 19
