"""Rebuild frozen MassBank inputs and evaluate complete cohort denominators."""
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import numpy as np

from .io import prepare_spectrum
from .output import exclusive_output


def verify_reproduction_file(path):
    path=Path(path)
    provenance=path.parent/'PROVENANCE.json'
    if provenance.exists():
        data=json.loads(provenance.read_text(encoding='utf-8'))
        expected=data.get('artifacts',{}).get(path.name)
        if expected is None or hashlib.sha256(path.read_bytes()).hexdigest()!=expected['sha256']:
            raise ValueError(f'Reproducibility artifact checksum mismatch: {path.name}')


def read_manifest(path):
    verify_reproduction_file(path)
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        rows=list(csv.DictReader(stream))
    ids=[r['accession'] for r in rows]
    if not rows or any(not i for i in ids) or len(ids)!=len(set(ids)):
        raise ValueError('Manifest must contain unique, nonempty accessions')
    for row in rows:
        if not row['connectivity_group'] or not row['source']:
            raise ValueError('Missing source or connectivity group')
        if row.get('fully_disjoint') not in ('True','False'):
            raise ValueError('Disjointness must be explicitly True or False')
    return rows


def record_input(row, source_root):
    root=Path(source_root).resolve()
    path=(root/row['relative_path']).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Record path escapes the source checkout')
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=row['source_sha256']:
        raise ValueError(f'Source checksum mismatch: {row["accession"]}')
    lines=raw.decode('utf-8-sig').splitlines()
    def field(prefix):
        values=[line[len(prefix):].strip() for line in lines if line.startswith(prefix)]
        if len(values)!=1:
            raise ValueError(f'Expected one {prefix} in {row["accession"]}')
        return values[0]
    if field('ACCESSION:')!=row['accession'] or field('LICENSE:')!=row['license']:
        raise ValueError('Record identity or license differs from manifest')
    mz=float(field('MS$FOCUSED_ION: PRECURSOR_M/Z '))
    if mz!=float(row['precursor_mz']) or field('MS$FOCUSED_ION: PRECURSOR_TYPE ')!='[M+H]+':
        raise ValueError('Precursor differs from frozen cohort')
    energy=float(row['nce_input'])
    if row['cohort']=='qtof_common':
        proxy=500.0*float(row['energy_numeric'])/mz
        if row['energy_basis'] not in ('explicit_eV','explicit_V','title_V'):
            raise ValueError('Unsupported QTOF energy basis')
        if float(np.float32(proxy))!=energy:
            raise ValueError('QTOF proxy NCE differs from recorded float32 input')
    elif row['cohort']=='orbitrap_six':
        if row['energy_basis']!='option_b_scalar_compatibility' or float(np.float32(float(row['energy_numeric'])))!=energy:
            raise ValueError('Orbitrap Option-B scalar differs from frozen float32 input')
    else:
        raise ValueError('Unknown cohort')
    starts=[i for i,line in enumerate(lines) if line.startswith('PK$PEAK:')]
    if len(starts)!=1:
        raise ValueError('Expected one peak table')
    peaks=[]
    for line in lines[starts[0]+1:]:
        if line.strip()=='//':
            break
        if line.strip():
            parts=line.split()
            if len(parts)<2:
                raise ValueError('Malformed MassBank peak row')
            peaks.append([float(parts[0]),float(parts[1])])
    if len(peaks)!=int(field('PK$NUM_PEAK:')):
        raise ValueError('Peak count does not match source record')
    record={'id':row['accession'],'precursor_mz':mz,'adduct':'[M+H]+',
            'collision_energy':energy,'peaks':peaks}
    prepared=prepare_spectrum(record)
    digest=hashlib.sha256(prepared.peaks.astype('<f4',copy=False).tobytes()).hexdigest()
    if digest!=row['preprocessed_peaks_sha256'] or prepared.peak_length!=int(row['retained_peaks']):
        raise ValueError(f'Preprocessing differs from frozen input: {row["accession"]}')
    return record


def rebuild(manifest, source_root, output):
    rows=read_manifest(manifest)
    with exclusive_output(output) as stream:
        for row in rows:
            record=record_input(row,source_root)
            stream.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')
    return len(rows)


def summarize_hits(rows, hits):
    views={'all':rows}
    for source in sorted({r['source'] for r in rows}):
        views['source:'+source]=[r for r in rows if r['source']==source]
    disjoint=[r for r in rows if r['fully_disjoint']=='True']
    if disjoint:
        views['internal_disjoint']=disjoint
        for source in sorted({r['source'] for r in disjoint}):
            views['internal_disjoint_source:'+source]=[r for r in disjoint if r['source']==source]
    result={}
    for name,group in views.items():
        by_connectivity=defaultdict(list)
        for row in group:
            by_connectivity[row['connectivity_group']].append(int(hits.get(row['accession'],False)))
        result[name]={'spectra':len(group),'connectivity_groups':len(by_connectivity),
                      'spectrum_top1':sum(sum(x) for x in by_connectivity.values())/len(group),
                      'molecule_top1':sum(sum(x)/len(x) for x in by_connectivity.values())/len(by_connectivity)}
    return result


def canonical_formula(formula):
    pieces=re.findall(r'([A-Z][a-z]?)([0-9]*)',formula)
    if not pieces or ''.join(e+n for e,n in pieces)!=formula:
        raise ValueError('Invalid elemental formula')
    counts=defaultdict(int)
    for element,number in pieces:
        count=int(number or 1)
        if count<1:
            raise ValueError('Nonpositive formula count')
        counts[element]+=count
    return tuple(sorted(counts.items()))


def evaluate_predictions(manifest, predictions):
    rows=read_manifest(manifest)
    known={r['accession']:r for r in rows};seen=set();hits={}
    with Path(predictions).open(encoding='utf-8-sig') as stream:
        for line in stream:
            if not line.strip():
                continue
            item=json.loads(line);accession=str(item['id'])
            if accession not in known or accession in seen:
                raise ValueError('Unknown or duplicate prediction id')
            seen.add(accession)
            first=[r for r in item.get('predictions',[]) if r.get('rank')==1]
            if len(first)>1:
                raise ValueError('More than one rank-1 prediction')
            hits[accession]=(item.get('status')=='ok' and bool(first) and
                             canonical_formula(first[0]['formula'])==canonical_formula(known[accession]['target_formula']))
    return {'method':'Spec2Formula','missing_predictions':len(rows)-len(seen),
            'failure_policy':'Missing, empty and unsuccessful outputs remain failures in all denominators',
            'metrics':summarize_hits(rows,hits)}


def evaluate_frozen_ranks(manifest, results):
    verify_reproduction_file(results)
    rows=read_manifest(manifest);known={r['accession'] for r in rows}
    with Path(results).open(encoding='utf-8-sig',newline='') as stream:
        ranks=list(csv.DictReader(stream))
    ids=[r['accession'] for r in ranks]
    if len(ids)!=len(set(ids)) or set(ids)!=known:
        raise ValueError('Frozen rank rows must match the complete manifest exactly')
    methods=['Spec2Formula','msbuddy','MIST_CF','SIRIUS']
    output={}
    for method in methods:
        hits={}
        for row in ranks:
            value=float(row[method+'_rank'] or 'nan')
            if math.isfinite(value) and value>0 and value!=int(value):
                raise ValueError('Positive target ranks must be integers')
            hits[row['accession']]=value==1
        output[method]=summarize_hits(rows,hits)
    return {'evaluation':'Recalculation of archived ranks; no new inference',
            'failure_policy':'Every manifest record retained; rank other than 1 contributes zero',
            'methods':output}
