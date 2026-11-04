"""Exit 0 if an idea-1 NCM summary has the planned settings: python -m code_scale.check_ncm_1 SUMMARY CODEC ENCODER [DEC_LAYERS]"""
import json
import sys

path, codec, enc = sys.argv[1:4]
dec = int(sys.argv[4]) if len(sys.argv) > 4 else 2
s = json.load(open(path))
ok = (s.get("codec") == codec and s.get("encoder") == enc and s.get("steps") == 1800 and s.get("reveal") == []
      and s.get("batch") == 32
      and s.get("attn_pattern") == "block_diagonal" and s.get("levels") == [4, 8, 16, 32, 48, 64, 64] and s.get("dec_layers", 2) == dec)
print("settings OK" if ok else f"settings differ: {s}")
sys.exit(0 if ok else 1)
