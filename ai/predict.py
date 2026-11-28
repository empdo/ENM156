import argparse
import json
from pathlib import Path
from model_utils import predict_file
from status_mapping import interpret_prediction


def main():
    parser = argparse.ArgumentParser(description="Klassificera WAV och tolka OK/FEL/INAKTIV")
    parser.add_argument("audio", type=Path)
    parser.add_argument("--model", type=Path,
                        default=Path(__file__).resolve().parent / "artifacts" / "model.joblib")
    args = parser.parse_args()
    result = predict_file(args.model, args.audio)
    result.update(interpret_prediction(result["prediction"]))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
