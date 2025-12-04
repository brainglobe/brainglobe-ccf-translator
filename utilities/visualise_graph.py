import os

import pandas as pd

base_path = "../brainglobe_ccf_translator"


def generate_mermaid(metadata):
    edges = set()
    for i, m in metadata.iterrows():
        s = m["source_space"] + "_P" + str(int(m["source_age_pnd"]))
        t = m["target_space"] + "_P" + str(int(m["target_age_pnd"]))
        # Sanitize node names for Mermaid (replace special chars)
        s = s.replace("-", "_").replace(" ", "_")
        t = t.replace("-", "_").replace(" ", "_")
        # Sort to ensure A---B and B---A are treated as the same edge
        edge = tuple(sorted([s, t]))
        edges.add(edge)

    edge_lines = [f"    {a} --- {b}" for a, b in sorted(edges)]
    mermaid = "```mermaid\ngraph TD\n" + "\n".join(edge_lines) + "\n```"
    return mermaid


key_ages = [56, 28, 21, 14, 7, 4]

# Example usage
metadata_path = os.path.join(base_path, "metadata", "translation_metadata.csv")
metadata = pd.read_csv(metadata_path)
# get rid of interpolated nodes
metadata = metadata.loc[
    metadata[metadata["key_age"]][
        ["source_space", "target_space", "source_key_age", "target_key_age"]
    ]
    .drop_duplicates()
    .index
]
demba = metadata[metadata["source_space"].str.contains("demba")]
other = metadata[~metadata["source_space"].str.contains("demba")]
demba = demba[
    demba["source_age_pnd"].isin(key_ages)
    & demba["target_age_pnd"].isin(key_ages)
]

metadata = pd.concat([demba, other])
mermaid_output = generate_mermaid(metadata)
print(mermaid_output)

# Optionally write to README
readme_path = os.path.join(os.path.dirname(__file__), "README.md")
with open(readme_path, "w") as f:
    f.write("# Translation Graph\n\n")
    f.write(
        "This diagram shows the available translation paths between coordinate spaces.\n\n"
    )
    f.write(mermaid_output)
