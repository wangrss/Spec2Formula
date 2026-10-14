"""Recalculate archived metrics or evaluate new predictions without dropping failures."""
import argparse
import json
from spec2formula.massbank import evaluate_predictions, evaluate_frozen_ranks
from spec2formula.output import exclusive_output

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',required=True)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--predictions')
    mode.add_argument('--frozen-ranks')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    report=(evaluate_predictions(args.manifest,args.predictions) if args.predictions
            else evaluate_frozen_ranks(args.manifest,args.frozen_ranks))
    with exclusive_output(args.output) as stream:
        json.dump(report,stream,indent=2,allow_nan=False)
        stream.write('\n')

if __name__=='__main__':
    main()
