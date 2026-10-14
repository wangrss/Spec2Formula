"""Frozen scientific constraints, separate from operational chunk settings."""
from .formula.elements import ELEMENTS, EXPANDED_ELEMENTS

PROTON_MASS_DA=1.007276466621
MAXIMUM_MZ=500.0
MAXIMUM_PEAKS=128
EVIDENCE_PEAKS=32
EVIDENCE_FEATURES=40
RERANK_TOP_K=20
DOMAIN_PPM=3.0
DOMAIN_RELATIVE_TOLERANCE=3.0e-6
DOMAIN_FLOOR_DA=.001
# Keep model feature scales and enumeration limits distinct even though this
# reference model uses the same numeric values for both.
COUNT_SCALES=(43,77,1,13,19,17,5,5,8,8,1,2,6,1,1,5)
ENUMERATION_LIMITS=(43,77,1,13,19,17,5,5,8,8,1,2,6,1,1,5)
FROZEN_DOMAIN={'elements':list(EXPANDED_ELEMENTS),'count_limits':list(ENUMERATION_LIMITS),
               'ppm':DOMAIN_PPM,'floor_da':DOMAIN_FLOOR_DA,'require_carbon':True,
               'minimum_dbe':0.0,'enforce_integer_dbe':True,'enforce_senior':True}


def validate_frozen_config(config):
    checks={'candidate_domain':FROZEN_DOMAIN,
            'scope':{'adduct':'[M+H]+','precursor_mz_exclusive_max':MAXIMUM_MZ,'precursor_error_cohort_filter':None}}
    for section,expected in checks.items():
        if config.get(section)!=expected:
            raise ValueError(f'{section} differs from the frozen reference protocol')
    for name,value in [('maximum_peaks',MAXIMUM_PEAKS),('evidence_peaks',EVIDENCE_PEAKS)]:
        if config.get('preprocessing',{}).get(name)!=value:
            raise ValueError(f'preprocessing.{name} must equal {value}')
    inference=config.get('inference',{})
    if inference.get('rerank_top_k')!=RERANK_TOP_K:
        raise ValueError('inference.rerank_top_k must equal 20')
    chunk=inference.get('base_candidate_chunk')
    if type(chunk) is not int or chunk<1:
        raise ValueError('inference.base_candidate_chunk must be a positive integer')
    if inference.get('base_standardization')!='population mean/SD across complete candidate list, SD floor 0.001':
        raise ValueError('Unsupported base-score standardization')
    if inference.get('evidence_token_dtype')!='float32':
        raise ValueError('Evidence tokens must use float32')
    model=config.get('models',{})
    for name,values in [('candidate_ranker',{'max_peaks':MAXIMUM_PEAKS,'max_mz':MAXIMUM_MZ,'candidate_feature_dim':25}),
                        ('evidence_reranker',{'candidate_feature_dim':25,'evidence_feature_dim':EVIDENCE_FEATURES,'base_spectrum_dim':512})]:
        actual=model.get(name,{}).get('config',{})
        if any(actual.get(key)!=value for key,value in values.items()):
            raise ValueError(f'{name} input dimensions differ from the frozen reference protocol')


def expected_native_protocol():
    return {'candidate_domain':FROZEN_DOMAIN,
            'exact_masses':[ELEMENTS[e].monoisotopic_mass for e in EXPANDED_ELEMENTS],
            'nominal_valences':[ELEMENTS[e].nominal_valence for e in EXPANDED_ELEMENTS]}
