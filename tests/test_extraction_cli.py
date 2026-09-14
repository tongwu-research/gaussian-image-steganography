import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from extraction_demo import make_example
from extract import extract

def test_receiver_process_reads_only_three_inputs(tmp_path):
    clean, embedded, profile = make_example(tmp_path)
    source = Path(__file__).resolve().parents[1] / 'src'
    result = subprocess.run([sys.executable,'-m','gaussian_steganography','extract','--clean',str(clean),'--embedded',str(embedded),'--map',str(profile)],cwd=tmp_path,env={**os.environ,'PYTHONPATH':str(source)},capture_output=True,text=True,check=True)
    assert json.loads(result.stdout)['bits'] == '01101111'
    assert set(p.name for p in tmp_path.iterdir()) == {'clean.pt','embedded.pt','decoder_map.json'}

def test_full_report_fields_are_rejected(tmp_path):
    clean, embedded, profile = make_example(tmp_path)
    data = json.loads(profile.read_text())
    data['assignments'][0]['selected_mask'] = [0,1,1,0]
    profile.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='Unexpected block fields'):
        extract(clean,embedded,profile)

def test_wrong_clean_file_is_rejected(tmp_path):
    clean, embedded, profile = make_example(tmp_path)
    data = torch.load(clean,weights_only=True)
    data['offsets'][0,0] = 0.1
    torch.save(data,clean)
    with pytest.raises(ValueError,match='does not match'):
        extract(clean,embedded,profile)

@pytest.mark.parametrize('score,expected',[(0.49,'0'),(0.50,'0'),(0.51,'1')])
def test_strict_action_threshold(tmp_path,score,expected):
    clean, embedded, profile = make_example(tmp_path)
    data = torch.load(embedded,weights_only=True)
    # Float64 and a binary-exact step make the equality case exact.
    data = {k:v.to(torch.float64) for k,v in data.items()}
    data['opacities'][0,0] = 0.5 + 0.125*score
    torch.save(data,embedded)
    mapping = json.loads(profile.read_text()); mapping['steps']['alpha_delta'] = 0.125
    profile.write_text(json.dumps(mapping))
    assert extract(clean,embedded,profile)['bits'][0] == expected
