"""Reconstruct exact paper inputs from a user-supplied MassBank checkout."""
import argparse
from spec2formula.massbank import rebuild

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',required=True)
    parser.add_argument('--source-root',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    print(f'Rebuilt {rebuild(args.manifest,args.source_root,args.output)} spectra')

if __name__=='__main__':
    main()
