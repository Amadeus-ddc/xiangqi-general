from pathlib import Path
import argparse
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .bridge import BoardLanguageModel
from .expert import FrozenPx0, encode_history
from .rules import START_FEN, replay
from .train import messages


def main():
    parser = argparse.ArgumentParser(description="Experimental stage-1 Xiangqi board QA; not a strong player")
    parser.add_argument("--adapter", default="runs/pilot-v1/adapter.pt")
    parser.add_argument("--weights", default="models/px0-latest.pb.gz")
    parser.add_argument("--initial-fen", default=START_FEN)
    parser.add_argument("--moves", nargs="*", default=[])
    parser.add_argument("--question", required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    history = replay(args.initial_fen, args.moves)
    record = {"fen": history[-1], "question": args.question}
    checkpoint = torch.load(args.adapter, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    expert = FrozenPx0(args.weights).cuda()
    with torch.no_grad():
        features, _ = expert(encode_history(history).unsqueeze(0).cuda())
    features = [f.to(torch.bfloat16) for f in features]
    del expert
    tokenizer = AutoTokenizer.from_pretrained(config["model_path"], local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(config["model_path"], local_files_only=True,
                                               dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
    model = BoardLanguageModel(base, features[0].shape[-1], config["decoder_bridge_positions"],
                              config["bridge_width"]).to("cuda", torch.bfloat16)
    model.bridges.load_state_dict(checkpoint["bridges"])
    model.eval()
    text = tokenizer.apply_chat_template(messages(record), tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to("cuda")
    with torch.no_grad(), model.board_context(features):
        result = model.base.generate(**inputs, do_sample=False, max_new_tokens=40,
                                     pad_token_id=tokenizer.pad_token_id)
    answer = tokenizer.decode(result[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    print(json.dumps({"fen": record["fen"], "question": args.question, "answer": answer,
                      "phase": "experimental_stage1", "external_oracle_used": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
