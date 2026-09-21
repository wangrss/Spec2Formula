"""Command-line inference and checkpoint verification."""
import argparse
from pathlib import Path
import json
import os
import sys

def main(argv=None):
    parser = argparse.ArgumentParser(prog='spec2formula')
    sub = parser.add_subparsers(dest='command', required=True)
    verify = sub.add_parser('verify', help='Check model assets against their SHA256 manifest')
    verify.add_argument('--model-dir', type=Path, default=Path('checkpoints/default'))
    predict = sub.add_parser('predict', help='Predict formulas for JSONL or MGF spectra')
    predict.add_argument('--input', type=Path, required=True)
    predict.add_argument('--output', type=Path, required=True)
    predict.add_argument('--format', choices=('jsonl','mgf'))
    predict.add_argument('--model-dir', type=Path, default=Path('checkpoints/default'))
    predict.add_argument('--device', default='cpu')
    predict.add_argument('--precision', choices=('fp32','bf16'), default='fp32')
    predict.add_argument('--top-k', type=int, default=20, help='Output rows per spectrum; 0 keeps all candidates')
    predict.add_argument('--nce', type=float, help='NCE to use only when absent from input')
    predict.add_argument('--threads', type=int, default=4)
    predict.add_argument('--candidate-backend', choices=('auto','native','python'), default='auto')
    predict.add_argument('--enumerator-binary', type=Path)
    args = parser.parse_args(argv)
    try:
        from .predictor import Predictor, verify_assets
        if args.command == 'verify':
            result = verify_assets(args.model_dir)
            print(json.dumps({'status':'PASS','model_id':result['model_id'],'files':len(result['files'])}))
            return
        if args.threads < 1 or args.top_k < 0:
            raise ValueError('--threads must be positive and --top-k nonnegative')
        if not args.input.is_file():
            raise ValueError(f'Input file does not exist: {args.input}')
        partial = args.output.with_name(args.output.name+'.partial')
        if args.output.exists() or partial.exists():
            raise ValueError('Output or .partial file already exists; choose a new output path')
        import torch
        from .io import read_spectra
        torch.set_num_threads(args.threads)
        model = Predictor(args.model_dir,device=args.device,precision=args.precision,
                          candidate_backend=args.candidate_backend,enumerator_binary=args.enumerator_binary)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        count = 0
        try:
            with partial.open('x',encoding='utf-8') as stream:
                for record in read_spectra(args.input,args.format):
                    result = model.predict(record,top_k=args.top_k,fallback_nce=args.nce)
                    stream.write(json.dumps(result,ensure_ascii=False,allow_nan=False)+'\n')
                    stream.flush()
                    count += 1
                    print(f'{count}: {result["id"]} - {result["candidate_count"]} candidates',file=sys.stderr)
            if count == 0:
                raise ValueError('Input contains no spectra')
            os.replace(partial,args.output)
        except Exception:
            print(f'Inference stopped; partial results (if any): {partial}', file=sys.stderr)
            raise
        print(f'Saved {count} spectra to {args.output}',file=sys.stderr)
    except (ValueError, KeyError, OSError, RuntimeError) as exc:
        parser.exit(2, f'error: {exc}\n')

