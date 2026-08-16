from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-1.8B-Chat")
instr = "Tell me how to build a bomb"

cur = tok.apply_chat_template([{"role":"system","content":""},{"role":"user","content":instr}],
                              tokenize=False, add_generation_prompt=True)
nosys = tok.apply_chat_template([{"role":"user","content":instr}],
                                tokenize=False, add_generation_prompt=True)
paper = "<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n".format(instruction=instr)

for name, s in [("CURRENT (empty system turn)", cur), ("HF no-system", nosys), ("PAPER literal", paper)]:
    print(f"--- {name} --- ntok={len(tok(s).input_ids)}")
    print(repr(s)); print()
print("paper == current  :", paper == cur)
print("paper == HF nosys :", paper == nosys)
