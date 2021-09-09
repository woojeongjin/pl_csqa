import argparse
import pytorch_lightning as pl

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--note", type=str, default=None)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--seed", type=int, default=9595)
    parser.add_argument("--test-run", action="store_true")
    parser.add_argument("--test_only", action="store_true")
    parser.add_argument("--distributed-backend", type=str, default="dp")
    parser.add_argument("--model-type", type=str, default="bert-base-uncased")
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--val-check-interval", type=float, default=0.9)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-nb-epochs", type=int, default=15)
    parser.add_argument("--min-nb-epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--adam-eps", type=float, default=1e-06)
    parser.add_argument("--warmup-steps", type=int, default=150)
    parser.add_argument('--load', type=str, default=None,
                        help='Load the model (usually the fine-tuned model).')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Load the model (usually the fine-tuned model).')
    parser.add_argument("--cl", action="store_true")
    parser.add_argument("--temp", type=float, default=0.05)
                        

    args = parser.parse_args()

    pl.utilities.seed.seed_everything(args.seed)

    return args