# 3_choose_harvest_years.py | Made by Claude guided by JFR | Created 2026-04-15
# Chooses representative harvest years using Faustmann-rule clustering, then writes a mapping table for simplified, 
# financially consistent downstream rotation-year bucketing choices.
from __future__ import annotations
import csv
from dataclasses import dataclass
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "Output"
DEFAULT_OUTPUT_CSV = OUTPUT_DIR / "choose_harvest_years_mapping.csv"

@dataclass(slots=True)
class RunConfig:
	k: int = 6
	maximum_harvest_year: int = 50
	minimum_harvest_year: int = 3
	discount_rate: float = 0.05
	include_sentinel: bool = True
	output_csv: Path = DEFAULT_OUTPUT_CSV

CONFIG = RunConfig(
	# k=5,
	# maximum_harvest_year=50,
	# minimum_harvest_year=3,
	# discount_rate=0.05,
	# include_sentinel=True,
	# output_csv=OUTPUT_DIR / "choose_harvest_years_mapping.csv",
)

def faustmann_hurdle(harvest_year: int, discount_rate: float) -> float:
	if harvest_year < 2: raise ValueError("harvest_year must be at least 2.")
	if discount_rate <= 0: raise ValueError("discount_rate must be positive.")
	return discount_rate / (1 - (1 + discount_rate) ** (-(harvest_year - 1)))

def build_candidate_years(minimum_harvest_year: int, maximum_harvest_year: int, include_sentinel: bool) -> list[int]:
	if minimum_harvest_year < 3: raise ValueError("minimum_harvest_year must be at least 3.")
	if maximum_harvest_year < minimum_harvest_year: raise ValueError("maximum_harvest_year must be at least minimum_harvest_year.")
	years = list(range(minimum_harvest_year, maximum_harvest_year + 1))
	if include_sentinel: years.append(maximum_harvest_year + 1)
	return years

def precompute_segment_costs(years: list[int], values: list[float]) -> tuple[list[list[float]], list[list[int]]]:
	n = len(years)
	costs = [[0.0]*n for _ in range(n)]
	representatives = [[0]*n for _ in range(n)]
	for start in range(n):
		for stop in range(start, n):
			best_cost = float("inf")
			best_rep = start
			for rep in range(start, stop + 1):
				rep_value = values[rep]
				cost = 0.0
				for index in range(start, stop + 1):
					delta = values[index] - rep_value
					cost += delta*delta
				if cost < best_cost:
					best_cost = cost
					best_rep = rep
			costs[start][stop] = best_cost
			representatives[start][stop] = best_rep
	return costs, representatives

def choose_harvest_years(k: int, maximum_harvest_year: int, minimum_harvest_year: int = 3, discount_rate: float = 0.05, include_sentinel: bool = True) -> tuple[list[int], list[dict[str, float | int]], float]:
	years = build_candidate_years(minimum_harvest_year, maximum_harvest_year, include_sentinel)
	if k <= 0: raise ValueError("k must be positive.")
	if k > len(years): raise ValueError("k cannot exceed the number of candidate harvest years.")
	values = [faustmann_hurdle(year, discount_rate) for year in years]
	costs, representatives = precompute_segment_costs(years, values)
	n = len(years)
	dp = [[float("inf")]*n for _ in range(k + 1)]
	prev = [[-1]*n for _ in range(k + 1)]
	for stop in range(n):
		dp[1][stop] = costs[0][stop]
	for clusters in range(2, k + 1):
		for stop in range(clusters - 1, n):
			for split in range(clusters - 2, stop):
				candidate_cost = dp[clusters - 1][split] + costs[split + 1][stop]
				if candidate_cost < dp[clusters][stop]:
					dp[clusters][stop] = candidate_cost
					prev[clusters][stop] = split
	segments = []
	clusters = k
	stop = n - 1
	while clusters >= 1:
		start = 0 if clusters == 1 else prev[clusters][stop] + 1
		segments.append((start, stop))
		stop = prev[clusters][stop]
		clusters -= 1
	segments.reverse()
	chosen_years = []
	mapping_rows = []
	for start, stop in segments:
		rep_index = representatives[start][stop]
		chosen_year = years[rep_index]
		chosen_years.append(chosen_year)
		chosen_value = values[rep_index]
		for index in range(start, stop + 1):
			delta = values[index] - chosen_value
			mapping_rows.append({
				"original_harvest_year": years[index],
				"chosen_harvest_year": chosen_year,
				"original_faustmann_hurdle": values[index],
				"chosen_faustmann_hurdle": chosen_value,
				"squared_error": delta*delta,
			})
	return chosen_years, mapping_rows, dp[k][n - 1]

def write_mapping_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with open(path, "w", newline="", encoding="utf-8") as handle:
		writer = csv.DictWriter(handle, fieldnames=[
			"original_harvest_year",
			"chosen_harvest_year",
			"original_faustmann_hurdle",
			"chosen_faustmann_hurdle",
			"squared_error",
		])
		writer.writeheader()
		writer.writerows(rows)

def main() -> None:
	chosen_years, mapping_rows, total_error = choose_harvest_years(
		k=CONFIG.k,
		maximum_harvest_year=CONFIG.maximum_harvest_year,
		minimum_harvest_year=CONFIG.minimum_harvest_year,
		discount_rate=CONFIG.discount_rate,
		include_sentinel=CONFIG.include_sentinel,
	)
	write_mapping_csv(CONFIG.output_csv, mapping_rows)
	print(f"Chosen harvest years: {chosen_years}", flush=True)
	print(f"Total squared Faustmann-rule error: {total_error:.12f}", flush=True)
	print(f"Wrote {CONFIG.output_csv}", flush=True)
	for row in mapping_rows:
		print(
			f"{row['original_harvest_year']:>3} -> {row['chosen_harvest_year']:>3} | "
			f"h={row['original_faustmann_hurdle']:.12f} | "
			f"h*={row['chosen_faustmann_hurdle']:.12f} | "
			f"se={row['squared_error']:.12f}",
			flush=True,
		)

if __name__ == "__main__":
	main()