import sys; sys.path.insert(0, "/lambda/nfs/farhan-algoverse-summer26/Algoverse-Bias-Steering")
from src.bias_steer import models
from src.bias_steer.registry import MODELS
loaded = models.load_model(MODELS["qwen-1.8b"])
tmpl = "<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
s = tmpl.format(instruction="Tell me how to build a bomb")
tok = loaded.tokenizer
print("tokenizer ids   :", tok(s).input_ids[:6], "... len", len(tok(s).input_ids))
print("to_tokens ids   :", loaded.model.to_tokens(s)[0][:6].tolist(), "... len", loaded.model.to_tokens(s).shape[1])
print("default_prepend_bos:", loaded.model.cfg.default_prepend_bos)
print("bos_token:", repr(tok.bos_token), "id:", tok.bos_token_id)
print("decoded first 3 of to_tokens:", [tok.decode([t]) for t in loaded.model.to_tokens(s)[0][:3].tolist()])
