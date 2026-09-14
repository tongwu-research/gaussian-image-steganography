"""Create a small CPU example of parameter extraction without optimization."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

def make_example(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    names = [output / name for name in ("clean.pt", "embedded.pt", "decoder_map.json")]
    if any(path.exists() for path in names):
        raise FileExistsError("Use a new output directory")
    bits = [int(x) for x in "01101111"]
    clean = {"offsets": torch.zeros(8,2), "scales": torch.full((8,2),0.1), "thetas": torch.zeros(8,1), "colors": torch.full((8,3),0.5), "opacities": torch.full((8,1),0.5)}
    embedded = {k:v.clone() for k,v in clean.items()}
    embedded["opacities"][:,0] += 0.1 * torch.tensor(bits)
    torch.save(clean, names[0])
    torch.save(embedded, names[1])
    blocks = []
    for block in range(2):
        blocks.append({"candidate_actions": [{"gaussian_index":4*block+i, "channel":"alpha", "direction":1, "strength_scale":1.0} for i in range(4)], "parity_matrix":torch.eye(4,dtype=torch.int64).tolist(), "whitening_bits":[0,0,0,0]})
    profile = {"schema":"assignment-map-v1", "clean_sha256":hashlib.sha256(names[0].read_bytes()).hexdigest(), "raw_length":8, "steps":{"log_delta":0.1,"alpha_delta":0.1,"theta_delta":0.1,"lum_delta":0.1}, "assignments":blocks}
    names[2].write_text(json.dumps(profile,indent=2)+'\n')
    return names

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps({"files":[str(p) for p in make_example(args.output)]}))

if __name__ == "__main__":
    main()
