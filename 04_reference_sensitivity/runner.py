"""
runner.py -- subspace-restriction rerun, resumable.

Ray is initialised here, before `driver` (and hence SRS_ray) is imported, so
that these settings win over any internal ray.init().  Two things matter:

  * _temp_dir must NOT be under /tmp.  On this instance /tmp is a tmpfs, so
    Ray's object spilling would consume RAM rather than relieve it -- and
    build_hamiltonian gates task submission on psutil.virtual_memory().percent,
    which counts tmpfs as used.  Spilling to tmpfs therefore raises the very
    gauge that gates submission, which livelocks the build.
  * object_store_memory is set explicitly so most objects stay in RAM and
    spilling is the exception rather than the rule.
"""

import os, json, pickle, warnings
warnings.filterwarnings('ignore')

import ray

SPILL = '/var/tmp/ray_spill'          # on /, a real disk -- never /tmp
os.makedirs(SPILL, exist_ok=True)

ray.init(
    _temp_dir='/var/tmp/ray_session',
    object_store_memory=int(40e9),
    _system_config={
        "object_spilling_config": json.dumps(
            {"type": "filesystem",
             "params": {"directory_path": SPILL}}),
    },
)

print('ray session dir :', ray.worker._global_node.get_session_dir_path(), flush=True)
print('object store    :',
      ray.cluster_resources().get('object_store_memory', 0) / 1e9, 'GB', flush=True)

import pandas as pd
import sys
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, ROOT)                          # srs/
sys.path.insert(0, os.path.join(ROOT, 'data'))    # mol_info
import mol_info
from srs.driver import run_molecule, to_rows, CSV_COLS

BASE = os.path.dirname(os.path.abspath(__file__))   # was '/home/ec2-user/SageMaker'
PKL = f'{BASE}/rerun.pkl'
CSV = f'{BASE}/subspace_results.csv'

NAMES = ['H2', 'HF', 'LiH', 'H2O', 'BeH2', 'O2', 'NH3', 'BH3', 'NO', 'N2',
         'CN', 'CH', 'OH', 'NH', 'NF', 'CNH', 'CO', 'CH4', 'H2O2', 'H2CO',
         'C2H2', 'NH2OH']
assert len(NAMES) == len(set(NAMES)) == 22
CHECK_HS = {'H2', 'LiH', 'OH'}

res = pickle.load(open(PKL, 'rb')) if os.path.exists(PKL) else {}
print(f'resuming with {len(res)} molecules already done', flush=True)

for n in NAMES:
    if n in res:
        print(f'{n} skipped', flush=True)
        continue
    res[n] = run_molecule(n, getattr(mol_info, n), check_hs=(n in CHECK_HS))
    with open(PKL, 'wb') as f:          # dump after every molecule
        pickle.dump(res, f)
    print(f'{n} done', flush=True)

pd.DataFrame(to_rows(res)).reindex(columns=CSV_COLS).to_csv(CSV, index=False)
print(f'wrote {CSV} ({len(res)} molecules)', flush=True)




# import warnings, pickle; warnings.filterwarnings('ignore')
# import pandas as pd
# import mol_info

# import os, json, ray
# os.makedirs('/var/tmp/ray_spill', exist_ok=True)
# ray.init(
#     _temp_dir='/var/tmp/ray_session',
#     object_store_memory=int(40e9),
#     _system_config={
#         "object_spilling_config": json.dumps(
#             {"type": "filesystem",
#              "params": {"directory_path": "/var/tmp/ray_spill"}}),
#     },
# )

# from driver import run_molecule, to_rows, CSV_COLS

# NAMES = ['H2','HF','LiH','H2O','BeH2','O2','NH3','BH3','NO','N2','CN','CH',
#          'OH','NH','NF','CNH','CO','CH4','H2O2','H2CO','C2H2','NH2OH']
# assert len(NAMES) == len(set(NAMES)) == 22
# CHECK_HS = {'H2', 'LiH', 'OH'}

# # res = {}
# # for n in NAMES:
# #     res[n] = run_molecule(n, getattr(mol_info, n), check_hs=(n in CHECK_HS))
# #     print(f'{n} done', flush=True)
# PKL = '/home/ec2-user/SageMaker/rerun.pkl'
# res = pickle.load(open(PKL, 'rb')) if os.path.exists(PKL) else {}
# for n in NAMES:
#     if n in res:
#         print(f'{n} skipped', flush=True); continue
#     res[n] = run_molecule(n, getattr(mol_info, n), check_hs=(n in CHECK_HS))
#     with open(PKL, 'wb') as f:
#         pickle.dump(res, f)
#     print(f'{n} done', flush=True)

# pd.DataFrame(to_rows(res)).reindex(columns=CSV_COLS) \
#   .to_csv('subspace_results.csv', index=False)
# with open('rerun.pkl', 'wb') as f:
#     pickle.dump(res, f)