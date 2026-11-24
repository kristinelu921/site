"""Complete the signed-scale check for the eleven currently selected recipes."""
from pathlib import Path
import copy,hashlib,json,sys
import numpy as np
from PIL import Image
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'signed_depth_diagnostic'
sys.path.insert(0,str(OLD))
import check as C
from signed_depth_metric import CONTRACT,depth_signed_scale
from check import L,M,PP,P,J,SITE
PINS={}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):PINS[str(p)]=sha(p);return json.loads(Path(p).read_text())
def image(root,desc):
 p=root/desc['path'];PINS[str(p)]=sha(p);assert PINS[str(p)]==desc['sha256']
 a=np.asarray(Image.open(p).convert('RGB'));assert a.shape==(256,256,3)
 assert hashlib.sha256(a.tobytes()).hexdigest()==desc['pixels_sha256'];return a
reference=C.frozen_signed_reference();C.fixtures(reference)
prior=read(OLD/'results.json')
catalog_path=ROOT/'catalog_before.json'
catalog=read(catalog_path if catalog_path.exists() else SITE/'dist/10-12-validations/best-results.json')
assert prior['status']=='PASS'
for p,h in read(OLD/'input_sha256.json').items():
 checked=catalog_path if p==str(SITE/'dist/10-12-validations/best-results.json') and catalog_path.exists() else Path(p)
 assert sha(checked)==h,(p,'changed diagnostic input')
(ROOT/'catalog_before.json').write_text(json.dumps(catalog,separators=(',',':'))+'\n')
L.PACKS=ROOT.parent/'genlv_reference_recovery_0046/pack'
_,_,meta,index=L.load_pack('bench_v2_val')
assert meta['pack_id']==catalog['pack_id']==prior['pack_id'] and meta['supervision_version']==6
canroot=ROOT.parent/'genlv_reference_recovery_0046/cpu_scoring_v2/bench_v2_val'
canonical=read(canroot/'scored_eval_v2/results.json')
vcroot=ROOT.parents[1]/'2026-10-30/recovery_followthrough_2040/scores/C0/units/relative_depth'
vc=read(vcroot/'scored_eval_v2/results.json')
# Authenticate the C0 recovery manifest; every published P2 query score must replay.
vc_manifest=read(vcroot/'scored_eval_v2/score_manifest.json')
assert vc_manifest['outputs']['results.json']==sha(vcroot/'scored_eval_v2/results.json')
assert vc_manifest['authoritative'] and vc_manifest['requested_contract_satisfied']
assert vc_manifest['metric_identity']['release_sha256']==catalog['metric_release_sha256']
lvroot=Path('/tmp/kristine_lavin_p8_completed_retry18/results/views/bench_v2_val')
lv=read(lvroot/'results.json')
rroot=ROOT.parent/'normal_pixel_audit_1300/recovered/relationadapter_p1'
relation=read(rroot/'scored_eval_v2/results.json')
lvm={'episodes':read(ROOT/'lvm/episodes.json')}
for p in (ROOT/'lvm').glob('metadata_*.json'):read(p)
pd_source=SITE/'dist/bench-v2-recovery/data.json';pd=read(pd_source)
pd_task=next(c for m in pd['models'] if m['key']=='prompt_diffusion' for c in m['rows'] if c['task']=='relative_depth')
rows=copy.deepcopy(prior['per_candidate']);summaries=copy.deepcopy(prior['summary'])
for model,doc,root in [('visualcloze',vc,vcroot),('lavin_dit',lv,lvroot),('relation_adapter',relation,ROOT/'relation_adapter'),('lvm',lvm,ROOT/'lvm'),('prompt_diffusion',None,None)]:
 recipe=next(m['default_recipe'] for m in catalog['models'] if m['key']==model)
 published=next(c for c in catalog['candidates'] if c['recipe']==recipe and c['task']=='relative_depth')
 for q in range(17):
  ref=canonical['episodes'][f'relative_depth@q{q}'];target=image(canroot,ref['artifacts']['query_target'])
  if doc:
   e=next(e for e in doc['episodes'].values() if e['task']=='relative_depth' and e['q']==q)
   assert e['query_record']==ref['query_record']
   for field in ['query_input','query_target']:assert e['artifacts'][field]['pixels_sha256']==ref['artifacts'][field]['pixels_sha256']
   desc=next(iter(e['artifacts']['readouts'].values()))['real'];pred=image(root,desc)
   demos=[index[('relative_depth',r)] for r in e['demo_records']]
  else:
   e=ref;pe=next(e for e in pd_task['queries'] if e['q']==q)
   p=(pd_source.parent/pe['images']['A1']).resolve();assert p.is_relative_to((SITE/'dist').resolve())
   pred=np.asarray(Image.open(p).convert('RGB'));assert pred.shape==(256,256,3)
   assert hashlib.sha256(pred.tobytes()).hexdigest()==p.stem;PINS[str(p)]=sha(p)
   qp=(pd_source.parent/pe['images']['query_input']).resolve();PINS[str(qp)]=sha(qp)
   query=np.asarray(Image.open(qp).convert('RGB'));assert hashlib.sha256(query.tobytes()).hexdigest()==ref['artifacts']['query_input']['pixels_sha256']
   desc={'sha256':sha(p)}
   # Relative-depth GT comes from the canonical query, independent of demo count.
   demos=[index[('relative_depth',r)] for r in ref['demo_records']]
  ix=index[('relative_depth',ref['query_record'])]
  ctx=L.build_ctx('relative_depth','bench_v2_val',meta['records'][ix],ix,demos,target)
  dataset=ref['query_source'];lo,hi=P.DEPTH_RANGE_M[dataset]
  with np.errstate(divide='ignore'):depth=1/ctx['gt_inv']
  scored=ctx['gt_scored'] & P.depth_eval_region(dataset,target.shape) & (depth>=lo) & (depth<=hi)
  decoded,invalid=PP.depth_grey(pred);kwargs={'min_inv':P.depth_min_inv(dataset,ctx)}
  before=M.depth_midas(decoded,invalid,ctx['gt_inv'],scored,**kwargs)
  after=depth_signed_scale(decoded,invalid,ctx['gt_inv'],scored,**kwargs)
  assert after==reference(decoded,invalid,ctx['gt_inv'],scored,**kwargs)
  assert before['coverage']==after['coverage']
  if after['scale']>=0:assert before==after
  else:assert before['scale']==0
  valid=scored & ~invalid;x=decoded[valid];y=ctx['gt_inv'][valid]
  if x.size>=2 and np.ptp(x)>0:
   slope=float(np.sum((x-x.mean())*(y-y.mean()))/np.sum((x-x.mean())**2));intercept=float(y.mean()-slope*x.mean())
   assert np.isclose(slope,after['scale'],rtol=1e-8,atol=1e-8)
   aligned=np.full(scored.sum(),kwargs['min_inv']);aligned[valid[scored]]=slope*x+intercept;aligned=np.maximum(aligned,kwargs['min_inv'])
   ratio=np.maximum(aligned/ctx['gt_inv'][scored],ctx['gt_inv'][scored]/aligned)
   assert float(np.mean(ratio<1.25))==after['delta1']
  expected=next(r for r in published['queries'] if r['q']==q)
  assert abs(before['delta1']-expected['score'])<1e-10,(model,q,before['delta1'],expected['score'])
  rows.append(dict(model=model,recipe=recipe,q=q,query_record=ref['query_record'],candidate=0,prediction_sha256=desc['sha256'],before=before,after=after))
 sub=[r for r in rows if r['model']==model]
 queries=[dict(q=r['q'],query_record=r['query_record'],before=r['before']['delta1'],after=r['after']['delta1']) for r in sub]
 s=dict(model=model,recipe=recipe,queries=queries,candidate_images=len(sub),before=float(np.mean([q['before'] for q in queries])),after=float(np.mean([q['after'] for q in queries])),negative_scales=sum(r['after']['scale']<0 for r in sub),remaining_zero_scales=sum(r['after']['scale']==0 for r in sub),wins=sum(q['after']>q['before']+1e-12 for q in queries),losses=sum(q['after']<q['before']-1e-12 for q in queries))
 summaries.append(s);print(json.dumps({k:v for k,v in s.items() if k!='queries'}),flush=True)
assert {s['model'] for s in summaries}=={m['key'] for m in catalog['models']}
assert len(rows)==306
for s in summaries:
 s['label']=next(c['label'] for c in catalog['candidates'] if c['recipe']==s['recipe'])
result=dict(status='PASS',contract=CONTRACT,pack_id=meta['pack_id'],supervision_version=6,base_metric_release_sha256=catalog['metric_release_sha256'],summary=summaries,per_candidate=rows,original_query_means_replayed=187,candidate_images=len(rows),independent_covariance_alignment_check=True,frozen_function_minus_one_branch_check=True,fixtures_pass=True,inference_runs=0,inference_metadata_added=False,sign_selected_using_query_ground_truth=True,script_sha256=sha(__file__),adapter_sha256=sha(OLD/'signed_depth_metric.py'),limitations=['Only the eleven previously selected complete inference recipes are rescored; alternatives remain accessible under the previous protocol.', 'JFLUX/TA reported reference has no verified raw predictions and is not ranked in the new depth row.', 'Analogist, LaVin-DiT and LVM use authenticated original producer predictions; PromptDiffusion uses published lossless pixel-hashed predictions. Every original query mean reproduces, but original scoring manifests were not recovered for these sources.'])
(ROOT/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
(ROOT/'input_sha256.json').write_text(json.dumps(PINS,indent=2)+'\n')
print(json.dumps(dict(status='PASS_ALL_11_SELECTED_RECIPES',query_means=187,candidate_images=len(rows))))
