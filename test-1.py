from transformers import AutoTokenizer, AutoModel
import torch

# torch.set_default_tensor_type('torch.cuda.FloatTensor')
# import fix
tokenizer = AutoTokenizer.from_pretrained(
    "THUDM/chatglm2-6b", trust_remote_code=True)
model = AutoModel.from_pretrained(
    "THUDM/chatglm2-6b", trust_remote_code=True, device='cuda')

model = model.eval()
round = 0
while True:
    round += 1
    print("ROUND: ", round)
    response, history = model.chat(tokenizer, "你好", history=[])
    print(response)
    response, history = model.chat(tokenizer, "晚上睡不着应该怎么办", history=history)
    print(response)
