#!/usr/bin/env python3
import json
import networkx as nx
import leidenalg
import igraph


def _assign_edge_weights(G):
    for u, v, data in G.edges(data=True):
        edge_type = data.get('type', 'unknown')
        
        if edge_type == 'inter-function':
            data['weight'] = 0.1
        elif edge_type == 'intra-function':
            data['weight'] = 2.0
        elif edge_type == 'non-call':
            data['weight'] = 1.0
        else:
            data['weight'] = 1.0


def collapse_leiden(G, resolution=0.05, weighted=True, partition_type='CPM'):
    # Map partition type strings to leidenalg classes
    partition_classes = {
        'CPM': leidenalg.CPMVertexPartition,
        'RB': leidenalg.RBConfigurationVertexPartition,
        'Modularity': leidenalg.ModularityVertexPartition,
        'Significance': leidenalg.SignificanceVertexPartition,
    }
    
    if partition_type not in partition_classes:
        raise ValueError(
            f"Unknown partition_type '{partition_type}'. "
            f"Choose from: {list(partition_classes.keys())}"
        )
    
    partition_class = partition_classes[partition_type]
    
    # Assign edge weights based on type for better community detection
    if weighted:
        _assign_edge_weights(G)
    
    # Create igraph from NetworkX directed graph with node attribute mapping
    nodes = list(G.nodes())
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    
    # Build edges with remapped indices
    edges = [(node_to_idx[u], node_to_idx[v]) for u, v in G.edges()]
    
    ig = igraph.Graph(len(nodes), edges, directed=True)
    ig.vs['node_id'] = nodes
    
    # Add edge weights to igraph
    edge_weights = []
    for u, v in G.edges():
        edge_data = G.get_edge_data(u, v) or {}
        weight = edge_data.get('weight', 1.0)
        edge_weights.append(weight)
    ig.es['weight'] = edge_weights
    
    # Apply Leiden algorithm with specified partition type
    partition = leidenalg.find_partition(
        ig,
        partition_class,
        weights='weight' if weighted else None,
        resolution_parameter=resolution
    )
    
    # Map igraph community membership back to NetworkX nodes
    node_to_community = {}
    for vertex in ig.vs:
        node_id = vertex['node_id']
        node_to_community[node_id] = partition.membership[vertex.index]
    
    # Get unique communities
    communities_list = [
        [node for node in G.nodes() if node_to_community.get(node) == comm_idx]
        for comm_idx in range(len(partition))
    ]
    
    # Create a mapping from node to its community index
    node_to_community = {}
    for comm_idx, comm in enumerate(communities_list):
        for node in comm:
            node_to_community[node] = comm_idx
    
    # Create collapsed graph
    collapsed_G = nx.DiGraph()
    
    # For each community, create a merged node
    for comm_idx, community_nodes in enumerate(communities_list):
        if len(community_nodes) == 0:
            continue
        
        # Aggregate node data from all nodes in the community
        community_nodes_list = list(community_nodes)
        first_node = community_nodes_list[0]
        first_node_data = G.nodes[first_node]
        
        # Merge attributes
        merged_label_parts = []
        merged_func_names = set()
        is_entry_point = False
        has_non_call_links = False
        
        for node in community_nodes_list:
            node_data = G.nodes[node]
            merged_label_parts.append(node_data.get('label', f'unknown @ {node}'))
            merged_func_names.add(node_data.get('func', 'unknown'))
            is_entry_point = is_entry_point or node_data.get('entry_point', False)
            has_non_call_links = has_non_call_links or node_data.get('non_call_links', False)
        
        # Create subgraph from community nodes
        subgraph = G.subgraph(community_nodes_list).copy()
        
        # Build LLM-friendly JSON with assembly and control flow
        blocks = []
        for node in community_nodes_list:
            node_data = G.nodes[node]
            block_info = {
                'id': hex(node) if isinstance(node, int) else str(node),
                'function': node_data.get('func', 'unknown'),
                'label': node_data.get('label', f'block @ {node}'),
                'entry_point': node_data.get('entry_point', False),
                'assembly': node_data.get('instrs', [])
            }
            blocks.append(block_info)
        
        edges = []
        for src, dst, data in subgraph.edges(data=True):
            edge_info = {
                'from': hex(src) if isinstance(src, int) else str(src),
                'to': hex(dst) if isinstance(dst, int) else str(dst),
                'type': data.get('type', 'unknown'),
                'conditional': data.get('conditional', False)
            }
            edges.append(edge_info)
        
        subgraph_json_data = {
            'blocks': blocks,
            'edges': edges
        }
        json_subgraph = json.dumps(subgraph_json_data, indent=2)
        
        # Merge func names for community-level func attribute
        merged_func = ' | '.join(sorted(merged_func_names))
        
        # Get color from first function in the community
        community_color = first_node_data.get('color', 'gray')
        
        # Store only community-relevant attributes
        community_data = {
            'label': f'Community {comm_idx}',
            'func': merged_func,
            'entry_point': is_entry_point,
            'non_call_links': has_non_call_links,
            'community_size': len(community_nodes),
            'community_nodes': community_nodes_list,
            'subgraph_json': json_subgraph,
            'subgraph_object': subgraph,
            'color': community_color,
        }
        
        # Add merged node (use community index as identifier)
        collapsed_G.add_node(comm_idx, **community_data)
    
    # Add edges between communities (avoid self-loops)
    added_edges = set()
    for src_node in G.nodes():
        for dst_node in G.successors(src_node):
            src_comm = node_to_community[src_node]
            dst_comm = node_to_community[dst_node]
            
            # Skip self-loops (edges within the same community)
            if src_comm != dst_comm:
                edge_key = (src_comm, dst_comm)
                if edge_key not in added_edges:
                    # Preserve edge attributes from the original graph
                    edge_data = G.get_edge_data(src_node, dst_node)
                    collapsed_G.add_edge(src_comm, dst_comm, **edge_data)
                    added_edges.add(edge_key)
    
    return collapsed_G, communities_list