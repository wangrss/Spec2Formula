"""JSONL/MGF readers and the frozen spectrum preprocessing rules."""
from dataclasses import dataclass
from pathlib import Path
import json
import math
import re
import numpy as np
from .formula_ranker.features import PROTON_MASS_DA

@dataclass(frozen=True)
class Spectrum:
    id: str
    precursor_mz: float
    collision_energy: float
    peaks: np.ndarray
    peak_length: int
    warnings: tuple[str, ...]

    @property
    def neutral_mass(self):
        return self.precursor_mz - PROTON_MASS_DA

def prepare_spectrum(record, *, fallback_nce=None, max_peaks=128):
    if not isinstance(record, dict):
        raise ValueError("Each spectrum must be a JSON object")
    mz = float(record["precursor_mz"])
    if not math.isfinite(mz) or not (PROTON_MASS_DA + 0.001 < mz < 500.0):
        raise ValueError("precursor_mz must be finite, > proton mass + 0.001 and < 500 Da")
    if record.get("adduct", "[M+H]+") != "[M+H]+":
        raise ValueError("This checkpoint supports [M+H]+ only")
    notes = list(record.get("_reader_warnings", []))
    nce = record.get("collision_energy", record.get("nce"))
    if nce is None:
        nce = fallback_nce
        if nce is not None:
            notes.append("Missing NCE filled from --nce")
    if nce is None:
        nce = 0.0
        notes.append("Missing NCE filled with 0; supply normalized collision energy when available")
    nce = float(nce)
    if not math.isfinite(nce) or nce < 0 or nce > np.finfo(np.float32).max:
        raise ValueError("NCE must be a finite nonnegative float32 value")
    if nce > 400:
        notes.append("NCE > 400 is clipped by the model metadata transformation")
    peaks = np.asarray(record["peaks"], dtype=np.float64)
    if peaks.ndim != 2 or peaks.shape[1] != 2:
        raise ValueError("peaks must be an N x 2 array of [m/z, intensity]")
    valid = np.isfinite(peaks).all(axis=1) & (peaks > 0).all(axis=1)
    rows = [tuple(row) for row in peaks[valid]]
    if not rows:
        raise ValueError("No finite positive peaks remain")
    if not valid.all():
        notes.append(f"Removed {int((~valid).sum())} nonfinite/nonpositive peaks")
    rows.sort(key=lambda item: item[1], reverse=True)  # stable ties, as in reference
    if len(rows) > max_peaks:
        notes.append(f"Kept the {max_peaks} most intense peaks from {len(rows)} valid peaks")
    rows = rows[:max_peaks]
    maximum = rows[0][1]
    rows = sorted(((mz, intensity / maximum) for mz, intensity in rows), key=lambda item:item[0])
    if any(mz > np.finfo(np.float32).max for mz, _ in rows):
        raise ValueError("Peak m/z cannot be represented in float32")
    array = np.zeros((max_peaks, 2), dtype=np.float32)
    array[:len(rows)] = rows
    return Spectrum(str(record.get("id", "spectrum")), mz, nce, array, len(rows), tuple(notes))

def _mgf_record(meta, peaks, index):
    if "PEPMASS" not in meta:
        raise ValueError(f"MGF spectrum {index}: missing PEPMASS")
    if "CHARGE" in meta and meta["CHARGE"].strip() not in ("1+", "+1", "1"):
        raise ValueError(f"MGF spectrum {index}: only charge +1 is supported")
    record = {"id":meta.get("TITLE", str(index)), "precursor_mz":float(meta["PEPMASS"].split()[0]),
              "adduct":meta.get("ADDUCT", meta.get("ION", "[M+H]+")), "peaks":peaks}
    energy = meta.get("NCE")
    if energy is not None:
        match = re.fullmatch(r"\s*([0-9]+(?:\.[0-9]+)?)\s*(?:%|NCE)?\s*", energy, re.I)
        if not match:
            raise ValueError(f"MGF spectrum {index}: invalid NCE value")
        record["collision_energy"] = float(match.group(1))
    elif "COLLISIONENERGY" in meta or "CE" in meta:
        record["_reader_warnings"] = ["MGF CE/COLLISIONENERGY is not assumed to be NCE; use NCE= or --nce"]
    return record

def read_spectra(path, input_format=None):
    path = Path(path)
    fmt = input_format or path.suffix.lstrip('.').lower()
    if fmt == 'jsonl':
        with path.open(encoding='utf-8-sig') as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                    if not isinstance(item, dict):
                        raise ValueError('Expected an object')
                    item.setdefault('id', str(line_number))
                    yield item
                except (ValueError, TypeError) as exc:
                    raise ValueError(f'JSONL line {line_number}: {exc}') from exc
    elif fmt == 'mgf':
        meta, peaks, index = None, [], 0
        with path.open(encoding='utf-8-sig') as stream:
            for line_number, line in enumerate(stream, 1):
                line = line.strip()
                if not line or line.startswith(('#', ';', '!')):
                    continue
                if line.upper() == 'BEGIN IONS':
                    if meta is not None:
                        raise ValueError(f'MGF line {line_number}: nested BEGIN IONS')
                    meta, peaks = {}, []
                elif line.upper() == 'END IONS':
                    if meta is None:
                        raise ValueError(f'MGF line {line_number}: unmatched END IONS')
                    index += 1
                    yield _mgf_record(meta, peaks, index)
                    meta = None
                elif meta is not None:
                    if '=' in line:
                        key, value = line.split('=', 1)
                        meta[key.strip().upper()] = value.strip()
                    else:
                        fields = line.split()
                        if len(fields) < 2:
                            raise ValueError(f'MGF line {line_number}: invalid peak')
                        peaks.append([float(fields[0]), float(fields[1])])
        if meta is not None:
            raise ValueError('MGF ended before END IONS')
    else:
        raise ValueError('Supported input formats: .jsonl and .mgf')

