import os

def reindex_and_deduplicate(input_path, output_path):
    edges = set()
    unique_nodes = set()

    # 1) Read original file
    with open(input_path, 'r') as f:
        for line in f:
            s, d = line.strip().split()
            s, d = int(s), int(d)
            # Skip self-loops if desired
            if s == d:
                continue
            # Use a normalized order (smallest first) to avoid duplicates like (3,2) vs (2,3)
            edge = tuple(sorted([s, d]))
            edges.add(edge)

    # 2) Collect unique node IDs
    for (src, dst) in edges:
        unique_nodes.add(src)
        unique_nodes.add(dst)

    # 3) Sort them and build a mapping old_id -> new_id (0..N-1)
    sorted_nodes = sorted(unique_nodes)
    node_id_map = {old_id: i for i, old_id in enumerate(sorted_nodes)}

    # 4) Write out edges with new IDs
    #    Each line: "new_src new_dst"
    with open(output_path, 'w') as fout:
        for (src, dst) in edges:
            new_src = node_id_map[src]
            new_dst = node_id_map[dst]
            fout.write(f"{new_src} {new_dst}\n")

    print(f"Done! Wrote reindexed edges to {output_path} with {len(node_id_map)} unique nodes.")

# Example usage:
if __name__ == "__main__":
    input_file = "./graphs/irvine.txt"
    output_file = "./graphs/irvine_reindexed.txt"
    reindex_and_deduplicate(input_file, output_file)
