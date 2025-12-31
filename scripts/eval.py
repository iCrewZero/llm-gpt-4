from inference.chat import chat
from tokenizer.tokenizer import Tokenizer
from model.gpt import GPT
import yaml, torch

cfg = yaml.safe_load(open("config/model.yaml"))
model = GPT(type("cfg", (), cfg)).cuda()
tok = Tokenizer("tokenizer/tokenizer.model")

print(chat(model, tok, [
    {"role": "system", "content": "You are a helpful AI"},
    {"role": "user", "content": "Explain transformers simply"}
]))
