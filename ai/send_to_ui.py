import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ui"))
from pi_server import StatusServer
from model_utils import predict_file
from status_mapping import interpret_prediction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", nargs="+", type=Path, help="En eller flera WAV-filer")
    parser.add_argument("--model", type=Path, default=HERE / "artifacts" / "model.joblib")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--interval", type=float, default=3)
    args = parser.parse_args()
    server = StatusServer(port=args.port)
    server.start()
    print(f"Öppna ui/main.html och anslut till ws://DATORNS-IP:{args.port}")
    try:
        for wav in args.audio:
            result = predict_file(args.model, wav)
            status = interpret_prediction(result["prediction"])
            print(json.dumps({"file":str(wav), **status, **result}, ensure_ascii=False))
            server.set_status(status["status"])
            time.sleep(args.interval)
        print("Klar. Statusservern är kvar tills Ctrl+C. Vänta med att ansluta UI om du kör bara en ljudfil.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


if __name__ == "__main__":
    main()
