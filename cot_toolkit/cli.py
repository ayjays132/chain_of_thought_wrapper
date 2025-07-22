import argparse
import logging
from . import ChainOfThoughtWrapper, validate_device_selection
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run the Chain-of-Thought wrapper from the command line")
    parser.add_argument("model", help="Model name or path")
    parser.add_argument("prompt", help="Prompt text to generate from")
    parser.add_argument("--device", default="cpu", help="Device to load the model on")
    parser.add_argument("--max-new-tokens", type=int, default=64, dest="max_new_tokens",
                        help="Maximum new tokens to generate")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    device = validate_device_selection(args.device)
    logger.info("Loading model %s", args.model)
    model = AutoModel.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    wrapper = ChainOfThoughtWrapper(model=model, processor=tokenizer, device=device)
    result = wrapper.generate(args.prompt, generation_params={"max_new_tokens": args.max_new_tokens})
    if result.get("final_answers"):
        print(result["final_answers"][0])
    elif result.get("full_texts"):
        print(result["full_texts"][0])
    else:
        print("No output generated")


if __name__ == "__main__":
    main()
