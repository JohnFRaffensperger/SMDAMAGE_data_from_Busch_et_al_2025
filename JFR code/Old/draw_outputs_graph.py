# draw_outputs_graph.py | Created 2026-04-08
# Draws graph structures from model outputs, linking fields and calculations to visualize dependencies, transformations, and downstream reporting relationships for debugging.
from __future__ import annotations
from collections import defaultdict, deque
import importlib
from pathlib import Path
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = Path(__file__).resolve().parents[2]
GRAPH_FILE = PROJECT_DIR / "Output" / "40_reports" / "graph_of_variable_names.txt"
SVG_FILE = PROJECT_DIR / "Output" / "40_reports" / "graph_of_variable_names.svg"


def resolve_graph_file(path: Path) -> Path:
	candidates = [
		path,
		PROJECT_DIR / "Output" / "40_reports" / "graph_of_variable_names.txt",
		SCRIPT_DIR / "graph_of_variable_names.txt",
		SCRIPT_DIR.parent / "graph_of_variable_names.txt",
		SCRIPT_DIR / "outputs_graph.txt",
		SCRIPT_DIR.parent / "outputs_graph.txt",
		PROJECT_DIR / "outputs_graph.txt",
		PROJECT_DIR / "JFR code" / "outputs_graph.txt",
		PROJECT_DIR / "JFR code" / "Old" / "outputs_graph.txt",
	]
	for candidate in candidates:
		if candidate.exists() and candidate.is_file(): return candidate
	found = list(PROJECT_DIR.rglob("graph_of_variable_names.txt")) + list(PROJECT_DIR.rglob("outputs_graph.txt"))
	if len(found) == 1: return found[0]
	if len(found) > 1:
		raise FileNotFoundError(
			"Multiple graph input files found. Set GRAPH_FILE explicitly to one of: "
			+ "; ".join(str(p) for p in found)
		)
	raise FileNotFoundError(
		"Could not find graph_of_variable_names.txt or outputs_graph.txt. Create graph_of_variable_names.txt in Output/40_reports/, "
		"or set GRAPH_FILE to the full path."
	)


def read_edges(path: Path) -> list[tuple[str, str]]:
	edges: list[tuple[str, str]] = []
	for line in path.read_text(encoding="utf-8").splitlines():
		line = line.strip()
		if not line: continue
		source, target = line.split("\t")
		edges.append((source, target))
	return edges


def topological_levels(edges: list[tuple[str, str]]) -> tuple[list[str], dict[str, int], dict[str, set[str]], dict[str, set[str]]]:
	nodes = sorted({node for edge in edges for node in edge})
	predecessors: dict[str, set[str]] = {node: set() for node in nodes}
	successors: dict[str, set[str]] = {node: set() for node in nodes}
	indegree = {node: 0 for node in nodes}
	for source, target in edges:
		if target not in successors[source]:
			successors[source].add(target)
			predecessors[target].add(source)
			indegree[target] += 1
	queue = deque(sorted(node for node in nodes if indegree[node] == 0))
	order: list[str] = []
	while queue:
		node = queue.popleft()
		order.append(node)
		for target in sorted(successors[node]):
			indegree[target] -= 1
			if indegree[target] == 0: queue.append(target)
	if len(order) != len(nodes): raise ValueError("graph input file is not acyclic")
	level = {node: 0 for node in nodes}
	for node in order:
		for target in successors[node]:
			level[target] = max(level[target], level[node] + 1)
	return order, level, predecessors, successors


def _linear_sum_assignment(cost_array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
	try:
		scipy_optimize = importlib.import_module("scipy.optimize")
		scipy_linear_sum_assignment = getattr(scipy_optimize, "linear_sum_assignment")
		rows, cols = scipy_linear_sum_assignment(cost_array)
		return np.asarray(rows, dtype=int), np.asarray(cols, dtype=int)
	except ModuleNotFoundError:
		# Fallback when scipy is unavailable in the local environment.
		row_count, col_count = cost_array.shape
		unused_cols = set(range(col_count))
		assignment_cols: list[int] = []
		for row in range(row_count):
			best_col = min(unused_cols, key=lambda col: float(cost_array[row, col]))
			assignment_cols.append(best_col)
			unused_cols.remove(best_col)
		row_indices = np.array(list(range(row_count)), dtype=int)
		col_indices = np.array(assignment_cols, dtype=int)
		return row_indices, col_indices


def _assign_rows(cost_matrix: list[list[float]]) -> list[int]:
	if not cost_matrix: return []
	if not cost_matrix[0]: return []
	cost_array = np.array(cost_matrix, dtype=float)
	row_indices, column_indices = _linear_sum_assignment(cost_array)
	assignment = [0]*len(cost_matrix)
	for row_idx, col_idx in zip(row_indices, column_indices): assignment[row_idx] = int(col_idx)
	return assignment


def layered_order(order: list[str], level: dict[str, int], predecessors: dict[str, set[str]], successors: dict[str, set[str]]) -> tuple[dict[int, list[str]], dict[str, float], int]:
	layers: dict[int, list[str]] = defaultdict(list)
	for node in order: layers[level[node]].append(node)
	for layer in layers: layers[layer].sort()
	max_rows = max(len(layer_nodes) for layer_nodes in layers.values())
	positions: dict[str, float] = {}
	for layer in layers:
		nodes = layers[layer]
		top_margin = (max_rows - len(nodes))/2.0
		for idx, node in enumerate(nodes): positions[node] = top_margin + idx
	max_layer = max(layers)
	for _ in range(12):
		for layer in range(0, max_layer + 1):
			nodes = layers[layer]
			if not nodes: continue
			rows_in_layer = len(nodes)
			top_margin = (max_rows - rows_in_layer)/2.0
			candidate_rows = [top_margin + k for k in range(rows_in_layer)]
			cost_matrix: list[list[float]] = []
			for node in nodes:
				row_costs: list[float] = []
				for row in candidate_rows:
					cost = 0.0
					for pred in predecessors[node]: cost += abs(positions[pred] - row)
					for succ in successors[node]: cost += abs(positions[succ] - row)
					row_costs.append(cost)
				cost_matrix.append(row_costs)
			assignment = _assign_rows(cost_matrix)
			for idx, node in enumerate(nodes): positions[node] = candidate_rows[assignment[idx]]
		for layer in range(max_layer, -1, -1):
			nodes = layers[layer]
			if not nodes: continue
			rows_in_layer = len(nodes)
			top_margin = (max_rows - rows_in_layer)/2.0
			candidate_rows = [top_margin + k for k in range(rows_in_layer)]
			cost_matrix = []
			for node in nodes:
				row_costs = []
				for row in candidate_rows:
					cost = 0.0
					for pred in predecessors[node]: cost += abs(positions[pred] - row)
					for succ in successors[node]: cost += abs(positions[succ] - row)
					row_costs.append(cost)
				cost_matrix.append(row_costs)
			assignment = _assign_rows(cost_matrix)
			for idx, node in enumerate(nodes): positions[node] = candidate_rows[assignment[idx]]
	for layer in layers:
		layers[layer].sort(key=lambda node: (positions[node], node))
	return dict(layers), positions, max_rows


def svg_text(text: str) -> str:
	return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def write_svg(edges: list[tuple[str, str]], layers: dict[int, list[str]], level: dict[str, int], row_position: dict[str, float], max_rows: int) -> None:
	SVG_FILE.parent.mkdir(parents=True, exist_ok=True)
	box_h = 28
	x_gap = 120
	y_gap = 28
	margin = 40
	row_pitch = box_h + y_gap
	max_level = max(level.values())
	char_w = 7.2
	node_w = {node: max(120, int(18 + char_w*len(node))) for layer_nodes in layers.values() for node in layer_nodes}
	node_x: dict[str, float] = {}
	node_y: dict[str, float] = {}
	canvas_h = 2*margin + max_rows*row_pitch
	for layer, nodes in layers.items():
		x = margin + layer*(max(node_w.values()) + x_gap)
		for node in nodes:
			node_x[node] = x + node_w[node]/2
			node_y[node] = margin + row_position[node]*row_pitch + box_h/2
	canvas_w = margin*2 + len(layers)*max(node_w.values()) + max(0, len(layers) - 1)*x_gap
	parts = [
		f'<svg xmlns="http://www.w3.org/2000/svg" width="{int(canvas_w)}" height="{int(canvas_h)}" viewBox="0 0 {int(canvas_w)} {int(canvas_h)}">',
		'<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#6b7280"/></marker></defs>',
		'<rect width="100%" height="100%" fill="white"/>'
	]
	for source, target in edges:
		x1 = node_x[source] + node_w[source]/2
		y1 = node_y[source]
		x2 = node_x[target] - node_w[target]/2
		y2 = node_y[target]
		parts.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="#9ca3af" stroke-width="1.2" marker-end="url(#arrow)"/>')
	for layer, nodes in layers.items():
		for node in nodes:
			x = node_x[node] - node_w[node]/2
			y = node_y[node] - box_h/2
			fill = "#dbeafe" if level[node] == max_level else "#f8fafc"
			parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" rx="8" ry="8" width="{node_w[node]}" height="{box_h}" fill="{fill}" stroke="#334155" stroke-width="1"/>')
			parts.append(f'<text x="{node_x[node]:.1f}" y="{node_y[node] + 0.5:.1f}" text-anchor="middle" dominant-baseline="middle" font-family="Segoe UI, Arial, sans-serif" font-size="12" fill="#111827">{svg_text(node)}</text>')
	parts.append('</svg>')
	SVG_FILE.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
	graph_file = resolve_graph_file(GRAPH_FILE)
	edges = read_edges(graph_file)
	order, level, predecessors, successors = topological_levels(edges)
	layers, row_position, max_rows = layered_order(order, level, predecessors, successors)
	write_svg(edges, layers, level, row_position, max_rows)
	print(f"Read {graph_file}")
	print(f"Wrote {SVG_FILE}")


if __name__ == "__main__":
	main()