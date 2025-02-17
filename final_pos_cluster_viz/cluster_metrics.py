import os
import math
import networkx as nx
import numpy as np
import chess
import chess.pgn
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from community.community_louvain import best_partition, modularity
from positional_graph import PositionalGraph, get_zone

# --- Helper Functions for Extended Metrics ---

def union_influence_graph(pos_graph):
    """
    Returns the union influence graph (white and black merged)
    from the provided PositionalGraph instance.
    (Note: influence edges now use a uniform weight of 1.)
    """
    white_inf = pos_graph.compute_influence_subgraph_by_color(chess.WHITE)
    black_inf = pos_graph.compute_influence_subgraph_by_color(chess.BLACK)
    return nx.compose(white_inf, black_inf)

def compute_extended_influence_metrics(graph):
    """
    Wraps compute_graph_metrics() output into the following keys:
      - Overall Component Metrics:
            • diameter             (from adjusted_diameter)
            • avg_harmonic_centrality  (from adjusted_centrality)
            • avg_clustering        (from adjusted_clustering)
            • fiedler              (from adjusted_fiedler)
            • harmonic_centrality_variance (from centrality_variance)
      - overall_entropy: the computed cross entropy
      - Aggregated Zone Metrics:
            • diameter, avg_harmonic_centrality, avg_clustering, fiedler
              (entropy and harmonic_centrality_variance are set to 0)
      - Zone Metrics: for each zone, keys as above with entropy from zone data.
    """
    base = compute_graph_metrics(graph)
    overall = base["overall_component_metrics"]
    aggregated = base.get("aggregated_zone_metrics", {})
    zones = base.get("zone_metrics", {})

    ext_overall = {
        "diameter": overall.get("adjusted_diameter", 0),
        "avg_harmonic_centrality": overall.get("adjusted_centrality", 0),
        "avg_clustering": overall.get("adjusted_clustering", 0),
        "fiedler": overall.get("adjusted_fiedler", 0),
        "harmonic_centrality_variance": overall.get("centrality_variance", 0)
    }
    overall_entropy = base.get("cross_entropy", 0)
    
    ext_agg = {
        "diameter": aggregated.get("adjusted_diameter", 0),
        "avg_harmonic_centrality": aggregated.get("adjusted_centrality", 0),
        "avg_clustering": aggregated.get("adjusted_clustering", 0),
        "fiedler": aggregated.get("adjusted_fiedler", 0),
        "entropy": 0,
        "harmonic_centrality_variance": 0
    }
    
    ext_zones = {}
    for zone, data in zones.items():
        ext_zones[zone] = {
            "diameter": data.get("adjusted_diameter", 0),
            "avg_harmonic_centrality": data.get("adjusted_centrality", 0),
            "avg_clustering": data.get("adjusted_clustering", 0),
            "fiedler": data.get("adjusted_fiedler", 0),
            "entropy": data.get("entropy", 0),
            "harmonic_centrality_variance": 0
        }
    
    return {
        "overall_component_metrics": ext_overall,
        "overall_entropy": overall_entropy,
        "aggregated_zone_metrics": ext_agg,
        "zone_metrics": ext_zones
    }

# --- Extended Metrics Functions ---
# (Basic metrics functions are defined below.)

def compute_fiedler_value(graph):
    if graph.number_of_nodes() == 0:
        return 0.0
    L = nx.laplacian_matrix(graph, weight="weight").todense()
    eigenvalues = np.linalg.eigvalsh(L)
    for val in sorted(eigenvalues):
        if val > 1e-6:
            return val
    return 0.0

def compute_mean_centrality(graph):
    if graph.number_of_nodes() == 0:
        return 0.0
    centrality = nx.harmonic_centrality(graph, distance='weight')
    return np.mean(list(centrality.values())) if centrality else 0.0

def compute_diameter(graph):
    try:
        return nx.diameter(graph)
    except nx.NetworkXError:
        return 0

def compute_clustering_coefficient(graph):
    """
    Computes the average size of the largest maximal clique each node belongs to.
    This measure is a proxy for higher order clustering: larger maximal cliques
    indicate stronger cohesive groups.
    """
    cliques = list(nx.find_cliques(graph))
    max_clique_size = {node: 0 for node in graph.nodes()}
    for clique in cliques:
        clique_size = len(clique)
        for node in clique:
            if clique_size > max_clique_size[node]:
                max_clique_size[node] = clique_size
    if len(max_clique_size) == 0:
        return 0.0
    return sum(max_clique_size.values()) / len(max_clique_size)

def compute_entropy(probabilities):
    entropy = 0.0
    for p in probabilities:
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy

def cluster_component(component):
    partition = best_partition(component)
    clusters = []
    for cid in set(partition.values()):
        nodes = [n for n, c in partition.items() if c == cid]
        clusters.append(component.subgraph(nodes).copy())
    return clusters

def compute_component_metrics(component):
    return {
        'fiedler': compute_fiedler_value(component),
        'centrality': compute_mean_centrality(component),
        'diameter': compute_diameter(component),
        'clustering': compute_clustering_coefficient(component),
        'size': component.number_of_nodes()
    }

def compute_adjusted_component_metrics(component, component_metrics, clusters):
    component_size = component.number_of_nodes()
    if component_size == 0:
        return {
            'adjusted_fiedler': 0,
            'adjusted_centrality': 0,
            'adjusted_diameter': 0,
            'adjusted_clustering': 0,
            'centrality_variance': 0,
            'cross_entropy_cluster': 0,
            'size': 0
        }
    cluster_metrics = [compute_component_metrics(c) for c in clusters]
    cluster_ratios = [c['size'] / component_size for c in cluster_metrics]
    
    adjusted_fiedler = component_metrics['fiedler'] * sum(r * c['fiedler'] for r, c in zip(cluster_ratios, cluster_metrics))
    
    adjusted_centrality = component_metrics['centrality'] * (1 + sum(
        r * (c['centrality'] - component_metrics['centrality'])**2 
        for r, c in zip(cluster_ratios, cluster_metrics)
    ))
    
    adjusted_diameter = max(
        component_metrics['diameter'],
        max((c['diameter'] for c in cluster_metrics), default=component_metrics['diameter'])
    )
    
    if component.number_of_edges() > 0:
        try:
            mod = modularity(best_partition(component), component)
        except Exception:
            mod = 0.0
    else:
        mod = 0.0
    adjusted_clustering = component_metrics['clustering'] * mod
    
    centrality_values = list(nx.harmonic_centrality(component, distance='weight').values())
    ddof = 1 if len(centrality_values) > 1 else 0
    if len(centrality_values) == 0:
        centrality_variance = 0.0
    else:
        total_variance = np.var(centrality_values, ddof=ddof)
        between_var = sum(r * (c['centrality'] - component_metrics['centrality'])**2 
                          for r, c in zip(cluster_ratios, cluster_metrics))
        centrality_variance = between_var / total_variance if total_variance > 0 else 0.0
    
    cross_entropy_cluster = compute_entropy(cluster_ratios)
    
    return {
        'adjusted_fiedler': adjusted_fiedler,
        'adjusted_centrality': adjusted_centrality,
        'adjusted_diameter': adjusted_diameter,
        'adjusted_clustering': adjusted_clustering,
        'centrality_variance': centrality_variance,
        'cross_entropy_cluster': cross_entropy_cluster,
        'size': component_size
    }

def compute_graph_metrics(graph):
    components = list(nx.connected_components(graph))
    component_subgraphs = [graph.subgraph(c).copy() for c in components]
    total_nodes = graph.number_of_nodes()
    
    if total_nodes == 0:
        return {
            'overall_component_metrics': {
                'adjusted_fiedler': 0,
                'adjusted_centrality': 0,
                'adjusted_diameter': 0,
                'adjusted_clustering': 0,
                'centrality_variance': 0,
                'cross_entropy_cluster': 0,
                'size': 0
            },
            'zone_metrics': {},
            'aggregated_zone_metrics': {},
            'cross_entropy': 0
        }
    
    overall_component_metrics = {
        'adjusted_fiedler': 0.0,
        'adjusted_centrality': 0.0,
        'adjusted_diameter': 0,
        'adjusted_clustering': 0.0,
        'centrality_variance': 0.0,
        'cross_entropy_cluster': 0.0,
        'size': 0
    }
    component_sizes = []
    cross_entropy_clusters_total = 0.0
    
    for component in component_subgraphs:
        component_size = component.number_of_nodes()
        comp_metrics = compute_component_metrics(component)
        clusters = cluster_component(component)
        adjusted_metrics = compute_adjusted_component_metrics(component, comp_metrics, clusters)
        weight = component_size / total_nodes
        overall_component_metrics['adjusted_fiedler'] += adjusted_metrics['adjusted_fiedler'] * weight
        overall_component_metrics['adjusted_centrality'] += adjusted_metrics['adjusted_centrality'] * weight
        overall_component_metrics['adjusted_diameter'] = max(overall_component_metrics['adjusted_diameter'], adjusted_metrics['adjusted_diameter'])
        overall_component_metrics['adjusted_clustering'] += adjusted_metrics['adjusted_clustering'] * weight
        overall_component_metrics['centrality_variance'] += adjusted_metrics['centrality_variance'] * weight
        overall_component_metrics['cross_entropy_cluster'] += adjusted_metrics['cross_entropy_cluster'] * weight
        overall_component_metrics['size'] += component_size
        component_sizes.append(component_size)
        cross_entropy_clusters_total += adjusted_metrics['cross_entropy_cluster'] * weight
    
    p_components = np.array(component_sizes) / total_nodes
    H_component = -np.sum([p * np.log2(p) for p in p_components if p > 0])
    
    combined_cross_entropy = H_component + cross_entropy_clusters_total
    
    zones_list = ['queenside', 'kingside', 'center']
    zone_metrics = {}
    for zone in zones_list:
        zone_nodes = [n for n in graph.nodes() if isinstance(n, str) and n in {chess.square_name(sq) for sq in chess.SQUARES} and get_zone(n) == zone]
        zone_subgraph = graph.subgraph(zone_nodes).copy()
        components_zone = list(nx.connected_components(zone_subgraph))
        zone_component_subgraphs = [zone_subgraph.subgraph(c).copy() for c in components_zone]
        zone_component_metrics = []
        zone_component_sizes = []
        for zc in zone_component_subgraphs:
            zc_size = zc.number_of_nodes()
            zc_metrics = compute_component_metrics(zc)
            zc_clusters = cluster_component(zc)
            zc_adjusted = compute_adjusted_component_metrics(zc, zc_metrics, zc_clusters)
            zone_component_metrics.append(zc_adjusted)
            zone_component_sizes.append(zc_size)
        total_zone_nodes = sum(zone_component_sizes)
        if total_zone_nodes == 0:
            zone_metrics[zone] = {
                'adjusted_fiedler': 0,
                'adjusted_centrality': 0,
                'adjusted_diameter': 0,
                'adjusted_clustering': 0,
                'size': 0,
                'entropy': 0
            }
        else:
            zone_diameter = max(m['adjusted_diameter'] for m in zone_component_metrics)
            zone_fiedler = sum(m['adjusted_fiedler'] * (s / total_zone_nodes) for m, s in zip(zone_component_metrics, zone_component_sizes))
            zone_centrality = sum(m['adjusted_centrality'] * (s / total_zone_nodes) for m, s in zip(zone_component_metrics, zone_component_sizes))
            zone_clustering = sum(m['adjusted_clustering'] * (s / total_zone_nodes) for m, s in zip(zone_component_metrics, zone_component_sizes))
            zone_entropy = compute_entropy([s / total_zone_nodes for s in zone_component_sizes if s > 0])
            zone_metrics[zone] = {
                'adjusted_fiedler': zone_fiedler,
                'adjusted_centrality': zone_centrality,
                'adjusted_diameter': zone_diameter,
                'adjusted_clustering': zone_clustering,
                'size': total_zone_nodes,
                'entropy': zone_entropy
            }
    
    aggregated_zone_metrics = {
        'adjusted_fiedler': 0.0,
        'adjusted_centrality': 0.0,
        'adjusted_diameter': 0,
        'adjusted_clustering': 0.0
    }
    total_zone_nodes_all = sum(zone_metrics[zone]['size'] for zone in zones_list)
    for zone in zones_list:
        size_ratio = zone_metrics[zone]['size'] / total_zone_nodes_all if total_zone_nodes_all > 0 else 0
        aggregated_zone_metrics['adjusted_fiedler'] += zone_metrics[zone]['adjusted_fiedler'] * size_ratio
        aggregated_zone_metrics['adjusted_centrality'] += zone_metrics[zone]['adjusted_centrality'] * size_ratio
        aggregated_zone_metrics['adjusted_diameter'] = max(aggregated_zone_metrics['adjusted_diameter'], zone_metrics[zone]['adjusted_diameter'])
        aggregated_zone_metrics['adjusted_clustering'] += zone_metrics[zone]['adjusted_clustering'] * size_ratio
    
    return {
        'overall_component_metrics': overall_component_metrics,
        'zone_metrics': zone_metrics,
        'aggregated_zone_metrics': aggregated_zone_metrics,
        'cross_entropy': combined_cross_entropy
    }

# --- Comprehensive Graph and Summary Table Generation ---

def generate_comprehensive_outputs(game_fens, output_dir, game_info):
    """
    Given a list of FEN strings (one per move), compute extended influence metrics at every move for:
      - The union influence graph (white and black merged)
      - White-only influence graph
      - Black-only influence graph
      
    Then produce two outputs:
      1. A single comprehensive interactive graph that contains six subplots (one per metric).
         In each subplot, exactly 15 unique traces are added representing the following 15 categories:
           • Overall – Union, Overall – White, Overall – Black
           • Aggregated – Union, Aggregated – White, Aggregated – Black
           • Queenside – Union, Queenside – White, Queenside – Black
           • Kingside – Union, Kingside – White, Kingside – Black
           • Center – Union, Center – White, Center – Black
         Each trace is assigned a legendgroup (with the same name across subplots) so that toggling a legend
         item in the shared legend shows/hides that category in every subplot simultaneously.
      2. A single comprehensive interactive summary table that shows, for every move, all metric values for
         every category.
         
    All output HTML files are saved in output_dir.
    
    Returns a dictionary of computed metrics.
    """
    moves = []
    # Define metric keys and labels.
    metric_keys = ["diameter", "avg_harmonic_centrality", "avg_clustering", "fiedler", "entropy", "harmonic_centrality_variance"]
    metric_labels = {
        "diameter": "Diameter",
        "avg_harmonic_centrality": "Average Harmonic Centrality",
        "avg_clustering": "Average Clustering Coefficient",
        "fiedler": "Fiedler Value",
        "entropy": "Component Entropy",
        "harmonic_centrality_variance": "Harmonic Centrality Variance"
    }
    
    # Initialize containers for overall metrics.
    overall_union = {k: [] for k in metric_keys}
    overall_white = {k: [] for k in metric_keys}
    overall_black = {k: [] for k in metric_keys}
    # Aggregated zone metrics.
    agg_union = {k: [] for k in metric_keys}
    agg_white = {k: [] for k in metric_keys}
    agg_black = {k: [] for k in metric_keys}
    # Zone-specific metrics.
    zones = ["queenside", "kingside", "center"]
    zone_union = {zone: {k: [] for k in metric_keys} for zone in zones}
    zone_white = {zone: {k: [] for k in metric_keys} for zone in zones}
    zone_black = {zone: {k: [] for k in metric_keys} for zone in zones}
    
    # Process each move.
    for move_index, fen in enumerate(game_fens, start=1):
        moves.append(move_index)
        board = chess.Board(fen)
        pos_graph = PositionalGraph(board)
        
        # Compute union influence metrics.
        union_inf = union_influence_graph(pos_graph)
        ext_union = compute_extended_influence_metrics(union_inf)
        comp_u = ext_union["overall_component_metrics"]
        overall_union["diameter"].append(comp_u.get("diameter"))
        overall_union["avg_harmonic_centrality"].append(comp_u.get("avg_harmonic_centrality"))
        overall_union["avg_clustering"].append(comp_u.get("avg_clustering"))
        overall_union["fiedler"].append(comp_u.get("fiedler"))
        overall_union["entropy"].append(ext_union.get("overall_entropy"))
        overall_union["harmonic_centrality_variance"].append(comp_u.get("harmonic_centrality_variance"))
        agg_u = ext_union["aggregated_zone_metrics"]
        for k in metric_keys:
            agg_union[k].append(agg_u.get(k))
        for zone in zones:
            for k in metric_keys:
                zone_union[zone][k].append(ext_union["zone_metrics"].get(zone, {}).get(k))
        
        # Compute white-only influence metrics.
        white_inf = pos_graph.compute_influence_subgraph_by_color(chess.WHITE)
        ext_white = compute_extended_influence_metrics(white_inf)
        comp_w = ext_white["overall_component_metrics"]
        overall_white["diameter"].append(comp_w.get("diameter"))
        overall_white["avg_harmonic_centrality"].append(comp_w.get("avg_harmonic_centrality"))
        overall_white["avg_clustering"].append(comp_w.get("avg_clustering"))
        overall_white["fiedler"].append(comp_w.get("fiedler"))
        overall_white["entropy"].append(ext_white.get("overall_entropy"))
        overall_white["harmonic_centrality_variance"].append(comp_w.get("harmonic_centrality_variance"))
        agg_w = ext_white["aggregated_zone_metrics"]
        for k in metric_keys:
            agg_white[k].append(agg_w.get(k))
        for zone in zones:
            for k in metric_keys:
                zone_white[zone][k].append(ext_white["zone_metrics"].get(zone, {}).get(k))
        
        # Compute black-only influence metrics.
        black_inf = pos_graph.compute_influence_subgraph_by_color(chess.BLACK)
        ext_black = compute_extended_influence_metrics(black_inf)
        comp_b = ext_black["overall_component_metrics"]
        overall_black["diameter"].append(comp_b.get("diameter"))
        overall_black["avg_harmonic_centrality"].append(comp_b.get("avg_harmonic_centrality"))
        overall_black["avg_clustering"].append(comp_b.get("avg_clustering"))
        overall_black["fiedler"].append(comp_b.get("fiedler"))
        overall_black["entropy"].append(ext_black.get("overall_entropy"))
        overall_black["harmonic_centrality_variance"].append(comp_b.get("harmonic_centrality_variance"))
        agg_b = ext_black["aggregated_zone_metrics"]
        for k in metric_keys:
            agg_black[k].append(agg_b.get(k))
        for zone in zones:
            for k in metric_keys:
                zone_black[zone][k].append(ext_black["zone_metrics"].get(zone, {}).get(k))
    
    # === Create Comprehensive Graph with 6 Subplots and 15 Unique Legend Groups ===
    fig = make_subplots(rows=2, cols=3, subplot_titles=[metric_labels[k] for k in metric_keys])
    
    overall_categories = [
        ("Overall – Union", overall_union),
        ("Overall – White", overall_white),
        ("Overall – Black", overall_black)
    ]
    aggregated_categories = [
        ("Aggregated – Union", agg_union),
        ("Aggregated – White", agg_white),
        ("Aggregated – Black", agg_black)
    ]
    zone_categories = []
    for zone in zones:
        zone_label = zone.capitalize()
        zone_categories.append((f"{zone_label} – Union", zone_union[zone]))
        zone_categories.append((f"{zone_label} – White", zone_white[zone]))
        zone_categories.append((f"{zone_label} – Black", zone_black[zone]))
    
    all_categories = overall_categories + aggregated_categories + zone_categories  # 3 + 3 + 9 = 15

    legend_styles = {
        "Overall – Union": {"color": "#1f77b4", "dash": "solid"},
        "Overall – White": {"color": "#ff7f0e", "dash": "solid"},
        "Overall – Black": {"color": "#2ca02c", "dash": "solid"},
        "Aggregated – Union": {"color": "#d62728", "dash": "dash"},
        "Aggregated – White": {"color": "#9467bd", "dash": "dash"},
        "Aggregated – Black": {"color": "#8c564b", "dash": "dash"},
        "Queenside – Union": {"color": "#e377c2", "dash": "dot"},
        "Queenside – White": {"color": "#7f7f7f", "dash": "dot"},
        "Queenside – Black": {"color": "#bcbd22", "dash": "dot"},
        "Kingside – Union": {"color": "#17becf", "dash": "dot"},
        "Kingside – White": {"color": "#aec7e8", "dash": "dot"},
        "Kingside – Black": {"color": "#ffbb78", "dash": "dot"},
        "Center – Union": {"color": "#98df8a", "dash": "dot"},
        "Center – White": {"color": "#ff9896", "dash": "dot"},
        "Center – Black": {"color": "#c5b0d5", "dash": "dot"},
    }
    
    legend_shown = {}
    
    for i, (row, col) in enumerate([(1,1), (1,2), (1,3), (2,1), (2,2), (2,3)]):
        metric_key = metric_keys[i]
        for (cat_name, cat_data) in all_categories:
            y_data = cat_data.get(metric_key)
            show_legend = False
            if cat_name not in legend_shown:
                show_legend = True
                legend_shown[cat_name] = True
            style = legend_styles.get(cat_name, {"color": "black", "dash": "solid"})
            fig.add_trace(go.Scatter(
                x=moves,
                y=y_data,
                mode="lines+markers",
                name=cat_name,
                line=dict(color=style["color"], dash=style["dash"], width=2),
                legendgroup=cat_name,
                showlegend=show_legend
            ), row=row, col=col)
        fig.update_xaxes(title_text="Move Number", row=row, col=col)
        fig.update_yaxes(title_text=metric_labels[metric_key], row=row, col=col)
    
    fig.update_layout(
        title=(f"<b>Extended Influence Metrics Over Game Moves</b><br>"
               f"White: {game_info.get('White', 'N/A')} | Black: {game_info.get('Black', 'N/A')} | "
               f"Date: {game_info.get('Date', 'N/A')} | Opening: {game_info.get('Opening', 'N/A')} | "
               f"Result: {game_info.get('Result', 'N/A')}"),
        height=950,
        width=1700,
        legend=dict(orientation="v", x=1.02, y=1, font=dict(size=12)),
        font=dict(family="Helvetica, Arial", size=14, color="black"),
        plot_bgcolor="whitesmoke",
        paper_bgcolor="white",
        margin=dict(l=50, r=200, t=100, b=50)
    )
    
    comp_graph_path = os.path.join(output_dir, "comprehensive_metrics_graph.html")
    fig.write_html(comp_graph_path)
    
    table_columns = ["Move"]
    table_data = [moves]
    for key, label in zip(metric_keys, [metric_labels[k] for k in metric_keys]):
        table_columns.extend([f"Overall {label} (Union)", f"Overall {label} (White)", f"Overall {label} (Black)"])
        table_data.extend([overall_union[key], overall_white[key], overall_black[key]])
        table_columns.extend([f"Aggregated {label} (Union)", f"Aggregated {label} (White)", f"Aggregated {label} (Black)"])
        table_data.extend([agg_union[key], agg_white[key], agg_black[key]])
        for zone in zones:
            zone_label = zone.capitalize()
            table_columns.extend([
                f"{zone_label} {label} (Union)",
                f"{zone_label} {label} (White)",
                f"{zone_label} {label} (Black)"
            ])
            table_data.extend([
                zone_union[zone][key],
                zone_white[zone][key],
                zone_black[zone][key]
            ])
    
    table_fig = go.Figure(data=[go.Table(
        header=dict(values=table_columns, fill_color='paleturquoise', align='left', font=dict(size=12)),
        cells=dict(values=table_data, fill_color='lavender', align='left', font=dict(size=12))
    )])
    comp_table_path = os.path.join(output_dir, "comprehensive_summary_table.html")
    table_fig.write_html(comp_table_path)
    table_fig.update_layout(title="<b>Comprehensive Summary Table (All Moves)</b>", height=900, width=1800)
    
    return {
        "moves": moves,
        "overall_union": overall_union,
        "overall_white": overall_white,
        "overall_black": overall_black,
        "agg_union": agg_union,
        "agg_white": agg_white,
        "agg_black": agg_black,
        "zone_union": zone_union,
        "zone_white": zone_white,
        "zone_black": zone_black
    }

##################################################################
def calc_diff(adv_value, def_value, higher_better=True):
    if def_value == 0:
        return 0
    if higher_better:
        return (adv_value - def_value) / def_value * 100
    else:
        return (def_value - adv_value) / def_value * 100

def classify_use_case(adv, def_, side="White"):
    diff_diameter = calc_diff(adv["diameter"], def_["diameter"], higher_better=False)
    diff_hc       = calc_diff(adv["avg_harmonic_centrality"], def_["avg_harmonic_centrality"], higher_better=True)
    diff_amcs     = calc_diff(adv["avg_clustering"], def_["avg_clustering"], higher_better=True)
    diff_fiedler  = calc_diff(adv["fiedler"], def_["fiedler"], higher_better=True)
    diff_entropy  = calc_diff(adv["entropy"], def_["entropy"], higher_better=False)
    diff_var      = calc_diff(adv["harmonic_centrality_variance"], def_["harmonic_centrality_variance"], higher_better=False)

    if (diff_diameter >= 20 and diff_hc >= 20 and diff_amcs >= 30 and
        diff_fiedler >= 30 and diff_entropy >= 30 and diff_var >= 30):
        return f"Dominant {side}"
    if (diff_diameter >= 30 and diff_hc >= 25 and diff_amcs >= 40 and
        diff_fiedler >= 40 and diff_entropy >= 40 and diff_var >= 40):
        return f"Stable Fortress {side}"
    if (5 <= diff_diameter <= 10 and 10 <= diff_hc <= 15 and 10 <= diff_amcs <= 20 and
        0 <= diff_fiedler <= 10 and diff_entropy >= 15 and 5 <= diff_var <= 15):
        return f"Balanced Advantage {side}"
    if (10 <= diff_diameter <= 15 and abs(diff_hc) < 5 and diff_amcs <= -15 and
        10 <= diff_fiedler <= 15 and abs(diff_entropy) < 5 and 10 <= diff_var <= 15):
        return f"Mixed Signals {side}"
    if (diff_diameter <= -20 and diff_hc >= 20 and diff_amcs <= -20 and
        abs(diff_fiedler) < 5 and diff_entropy <= -20 and diff_var <= -20):
        return f"Overextended Attack {side}"
    if (diff_hc >= 30 and diff_fiedler <= -20 and diff_amcs <= -20 and
        diff_entropy >= 20 and diff_var >= 20 and 0 <= diff_diameter <= 10):
        return f"Concentrated but Fragile Initiative {side}"
    if (10 <= diff_diameter <= 15 and abs(diff_hc) < 5 and diff_amcs <= -30 and
        10 <= diff_fiedler <= 15 and -5 <= diff_entropy < 0 and 10 <= diff_var <= 15):
        return f"Fragmented Coordination {side}"
    if (abs(diff_diameter) < 5 and abs(diff_hc) < 5 and 5 <= diff_amcs <= 10 and
        0 <= diff_fiedler <= 10 and 5 <= diff_entropy <= 10 and abs(diff_var) < 5):
        return f"Dynamic Imbalance {side}"
    if (diff_diameter <= -15 and diff_hc >= 20 and diff_amcs <= -20 and
        diff_fiedler <= -5 and diff_entropy <= -20 and diff_var >= 20):
        return f"Ambitious but Uncoordinated {side}"
    if (diff_diameter >= 30 and -5 <= diff_hc <= 5 and 0 <= diff_amcs <= 10 and
        10 <= diff_fiedler <= 20 and diff_entropy >= 20 and diff_var >= 30):
        return f"Resilient but Limited {side}"
    if (abs(diff_diameter) < 5 and 5 <= diff_hc <= 10 and 0 <= diff_amcs <= 10 and
        abs(diff_fiedler) < 5 and diff_entropy >= 30 and diff_var >= 30):
        return f"Distributed Harmony {side}"
    return f"Unclassified {side}"

def generate_combined_use_case_plot(metrics, output_dir):
    moves = metrics["moves"]
    metric_keys = ["diameter", "avg_harmonic_centrality", "avg_clustering", "fiedler", "entropy", "harmonic_centrality_variance"]

    traces = []

    overall_white_cases = []
    overall_black_cases = []
    for i in range(len(moves)):
        white_move = { key: metrics["overall_white"][key][i] for key in metric_keys }
        black_move = { key: metrics["overall_black"][key][i] for key in metric_keys }
        overall_white_cases.append(classify_use_case(white_move, black_move, side="White"))
        overall_black_cases.append(classify_use_case(black_move, white_move, side="Black"))
    traces.append(("Overall – White", overall_white_cases, "blue"))
    traces.append(("Overall – Black", overall_black_cases, "red"))

    agg_white_cases = []
    agg_black_cases = []
    for i in range(len(moves)):
        white_move = { key: metrics["agg_white"][key][i] for key in metric_keys }
        black_move = { key: metrics["agg_black"][key][i] for key in metric_keys }
        agg_white_cases.append(classify_use_case(white_move, black_move, side="White"))
        agg_black_cases.append(classify_use_case(black_move, white_move, side="Black"))
    traces.append(("Aggregated – White", agg_white_cases, "darkblue"))
    traces.append(("Aggregated – Black", agg_black_cases, "darkred"))

    zones = ["queenside", "kingside", "center"]
    zone_colors = {"queenside": "green", "kingside": "purple", "center": "orange"}
    for zone in zones:
        zone_white_cases = []
        zone_black_cases = []
        for i in range(len(moves)):
            white_move = { key: metrics["zone_white"][zone][key][i] for key in metric_keys }
            black_move = { key: metrics["zone_black"][zone][key][i] for key in metric_keys }
            zone_white_cases.append(classify_use_case(white_move, black_move, side="White"))
            zone_black_cases.append(classify_use_case(black_move, white_move, side="Black"))
        traces.append((f"{zone.capitalize()} – White", zone_white_cases, zone_colors[zone]))
        traces.append((f"{zone.capitalize()} – Black", zone_black_cases, zone_colors[zone]))

    fig = go.Figure()
    for trace_name, trace_data, color in traces:
        fig.add_trace(go.Scatter(
            x=moves,
            y=trace_data,
            mode="lines+markers",
            name=trace_name,
            line=dict(color=color),
            marker=dict(size=8)
        ))
    fig.update_layout(
        title="Combined Use Case Classification Over Game Moves (All Categories)",
        xaxis_title="Move Number",
        yaxis_title="Use Case",
        yaxis=dict(type="category"),
        width=1200,
        height=800,
        template="plotly_white"
    )
    output_file = os.path.join(output_dir, "combined_use_case_timeseries.html")
    parent_dir = os.path.dirname(output_file)
    os.makedirs(parent_dir, exist_ok=True)
    
    fig.write_html(output_file)
    print(f"Combined use case time series plot saved to: {output_file}")

# --- Main Usage: Process All Games in the PGN File ---

if __name__ == "__main__":
    pgn_path = '/Users/ashutoshganguly/Desktop/Research Papers/abid_ashutosh_papers/chess_board_distance/try_8/freestyle_chess_games/carlsen_keymer_2025.pgn'
    output_base = '/Users/ashutoshganguly/Desktop/Research Papers/abid_ashutosh_papers/chess_board_distance/try_8/freestyle_chess_games'
    
    with open(pgn_path) as pgn_file:
        game_counter = 1
        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break
            game_fens = []
            board = game.board()
            game_fens.append(board.fen())
            for move in game.mainline_moves():
                board.push(move)
                game_fens.append(board.fen())
            
            game_folder = os.path.join(output_base, f'Game_{game_counter}')
            os.makedirs(game_folder, exist_ok=True)
            metrics = generate_comprehensive_outputs(game_fens, game_folder, game.headers)
            output_dir = os.getcwd()  
            generate_combined_use_case_plot(metrics, game_folder)
            print(f"Processed Game {game_counter}: {game.headers.get('White', 'N/A')} vs {game.headers.get('Black', 'N/A')}")
            game_counter += 1
