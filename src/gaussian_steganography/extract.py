"""Extract bits using an encoder-supplied assignment map."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from stego_protocol_v3 import decode_cost_code_v3

MAP_FIELDS = {"schema", "clean_sha256", "raw_length", "steps", "assignments"}
ACTION_FIELDS = {"gaussian_index", "channel", "direction", "strength_scale"}
BLOCK_FIELDS = {"candidate_actions", "parity_matrix", "whitening_bits"}
PARAMETERS = {"offsets": 2, "scales": 2, "thetas": 1, "colors": 3, "opacities": 1}

def load_parameters(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    params = payload.get("parameters", payload)
    count = None
    for name, width in PARAMETERS.items():
        value = params[name]
        if not isinstance(value, torch.Tensor) or value.ndim != 2 or value.shape[1] != width:
            raise ValueError(f"Invalid parameter shape: {name}")
        if not torch.isfinite(value).all():
            raise ValueError(f"Non-finite parameter: {name}")
        count = value.shape[0] if count is None else count
        if value.shape[0] != count:
            raise ValueError("Parameter counts differ")
    if not count or not (params["scales"] > 0).all():
        raise ValueError("Empty parameters or non-positive scales")
    return {name: params[name] for name in PARAMETERS}

def validate_map(profile, count):
    if set(profile) != MAP_FIELDS or profile["schema"] != "assignment-map-v1":
        raise ValueError("Expected a restricted assignment-map-v1 file, not an experiment report")
    if set(profile["steps"]) != {"log_delta", "alpha_delta", "theta_delta", "lum_delta"}:
        raise ValueError("All four channel step sizes must be specified")
    if any(not isinstance(x, (int, float)) or not 0 < x < float("inf") for x in profile["steps"].values()):
        raise ValueError("Step sizes must be finite and positive")
    blocks = profile["assignments"]
    if not isinstance(profile["raw_length"], int) or not 0 < profile["raw_length"] <= 4 * len(blocks):
        raise ValueError("Invalid payload length")
    for block in blocks:
        if set(block) != BLOCK_FIELDS:
            raise ValueError("Unexpected block fields")
        actions = block["candidate_actions"]
        matrix = block["parity_matrix"]
        if not actions or len(matrix) != len(actions) or any(len(row) != 4 or any(x not in (0,1) for x in row) for row in matrix):
            raise ValueError("Parity must be an action-by-four binary matrix")
        if len(block["whitening_bits"]) != 4 or any(x not in (0,1) for x in block["whitening_bits"]):
            raise ValueError("Whitening must contain four bits")
        for action in actions:
            if set(action) != ACTION_FIELDS:
                raise ValueError("Unexpected action fields")
            i = action["gaussian_index"]
            if not isinstance(i, int) or not 0 <= i < count:
                raise ValueError("Carrier index is outside the parameter array")
            if action["channel"] not in {"log_anisotropy", "alpha", "theta", "color_lum"} or action["direction"] not in (-1,1):
                raise ValueError("Invalid channel or direction")
            if not 0 < action["strength_scale"] < float("inf"):
                raise ValueError("Invalid action strength")

def extract(clean_path, embedded_path, map_path):
    clean_path = Path(clean_path)
    profile = json.loads(Path(map_path).read_text())
    clean = load_parameters(clean_path)
    embedded = load_parameters(embedded_path)
    if embedded["offsets"].shape != clean["offsets"].shape:
        raise ValueError("Clean and embedded parameter counts differ")
    validate_map(profile, len(clean["offsets"]))
    if hashlib.sha256(clean_path.read_bytes()).hexdigest() != profile["clean_sha256"]:
        raise ValueError("The clean file does not match the supplied map")
    bits, probabilities, scores = decode_cost_code_v3(embedded, clean_params=clean, assignments=profile["assignments"], raw_length=profile["raw_length"], channel_weights={}, **profile["steps"])
    return {"bits": "".join(map(str, bits.tolist())), "length": len(bits), "block_probabilities": probabilities, "action_scores": scores}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean", required=True, type=Path)
    parser.add_argument("--embedded", required=True, type=Path)
    parser.add_argument("--map", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(extract(args.clean, args.embedded, args.map)))

if __name__ == "__main__":
    main()
