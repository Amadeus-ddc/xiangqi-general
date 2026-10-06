import argparse
import json
import torch
from .rules import START_FEN, replay
from .inference import Predictor


def main():
    parser = argparse.ArgumentParser(description="Xiangqi General inference with complete game history")
    parser.add_argument("--adapter", default="runs/pilot-v1/adapter.pt")
    parser.add_argument("--weights", default="models/px0-latest.pb.gz")
    parser.add_argument("--initial-fen", default=START_FEN)
    parser.add_argument("--moves", nargs="*", default=[])
    parser.add_argument("--question", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    args = parser.parse_args()
    torch.set_num_threads(8)
    history = replay(args.initial_fen, args.moves)
    record = {"fen": history[-1], "history": history, "question": args.question}
    predictor = Predictor(args.adapter, args.weights)
    answer = predictor.generate(record, max_new_tokens=args.max_new_tokens)
    print(json.dumps({"fen": record["fen"], "question": args.question, "answer": answer,
                      "checkpoint": args.adapter, "mode": predictor.config.get('mode', 'bridge'),
                      "external_oracle_used": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
