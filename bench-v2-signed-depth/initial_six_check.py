"""Compare current depth decoding with/without the negative-scale reset."""
from pathlib import Path
import ast
import hashlib
import inspect
import json
import sys
import numpy as np
from PIL import Image
from signed_depth_metric import CONTRACT, depth_signed_scale

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
RUNS=ROOT.parent
REPO=ROOT.parents[2]
SITE=REPO/'site_workspaces/visualcloze-evals-demo-tables-20261026'
J=Path('/kmh-nfs-ssd-us-mount/code/xtiange/jflux2')
sys.path[:0]=[str(SITE/'scripts'),str(J)]
from depth_output_adapter import decode_depth_prediction
from tools import eval_v2_score_v1 as L
from jflux2.fasteval.eval_v2 import metrics_v1 as M,postprocess_v2 as PP,protocol_v1 as P
PINS={}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):PINS[str(p)]=sha(p);return json.loads(Path(p).read_text())
def write(name,d):(ROOT/name).write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def image(root,desc):
    p=root/desc['path'];assert sha(p)==desc['sha256'];PINS[str(p)]=sha(p)
    a=np.asarray(Image.open(p).convert('RGB'));assert a.shape==(256,256,3)
    assert hashlib.sha256(a.tobytes()).hexdigest()==desc['pixels_sha256'];return a


def frozen_signed_reference():
    # Independently execute the frozen function with precisely its s<0 branch
    # removed. Check every output against the readable local variant.
    node=ast.parse(inspect.getsource(M.depth_midas)).body[0]
    remove=[n for n in node.body if isinstance(n,ast.If) and ast.unparse(n.test)=='s < 0']
    assert len(remove)==1
    node.body.remove(remove[0]);node.name='reference_signed'
    module=ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[]));ns={'np':np}
    exec(compile(module,'<frozen depth_midas without negative reset>','exec'),ns)
    return ns['reference_signed']


def fixtures(reference):
    y=np.array([[.2,.4,.6,.8]])
    valid=np.ones_like(y,bool);no=np.zeros_like(y,bool)
    # Correct order; reversed order; flat output; missing prediction.
    for x,invalid in [(y,no),(1-y,no),(np.ones_like(y)*.5,no),(y,valid)]:
        s=depth_signed_scale(x,invalid,y,valid,min_inv=.1)
        assert s==reference(x,invalid,y,valid,min_inv=.1)
    assert depth_signed_scale(1-y,no,y,valid,min_inv=.1)['delta1']==1
    assert M.depth_midas(1-y,no,y,valid,min_inv=.1)['scale']==0
    assert depth_signed_scale(y,valid,y,valid,min_inv=.1)['coverage']==0
    assert depth_signed_scale(y,valid,y,valid,min_inv=.1)['delta1']==0


def main():
    reference=frozen_signed_reference();fixtures(reference)
    catalog=read(SITE/'dist/10-12-validations/best-results.json');write('catalog_before.json',catalog)
    defaults={m['key']:m['default_recipe'] for m in catalog['models']}
    units=[('painter',RUNS/'painter_reference_recovery_0420/archive_scoring/bench_v2_val'),
        ('vict',RUNS/'depth_convention_diagnostic/vict'),
        ('genlv',RUNS/'genlv_reference_recovery_0046/cpu_scoring_v2/bench_v2_val'),
        ('improv',RUNS/'improv_reference_recovery_0203/cpu_scoring_v2/bench_v2_val'),
        ('instamanip',Path('/tmp/instamanip-p3-score-certified-20261023T053423Z-628874/corrected_v3/validation/units/bench_v2_val'))]
    write('plan.json',dict(contract=CONTRACT,models=[m for m,_ in units]+['analogist'],
        change='Remove only the negative fitted scale → zero reset.',
        retained=['Current selected recipe','Current model-specific decoder','All eight Analogist candidates',
            'Target/crop/range','Missing-prediction mask and penalties','Minimum inverse depth','Metric formulas'],
        ground_truth='Query ground truth determines the affine scale, including its sign. This is evaluation, not target-free postprocessing.',
        main_table_changed=False,inference_runs=0))
    L.PACKS=RUNS/'genlv_reference_recovery_0046/pack'
    _,_,meta,index=L.load_pack('bench_v2_val')
    assert meta['pack_id']==catalog['pack_id'] and meta['supervision_version']==6
    docs={}
    for model,root in units:
        d=read(root/'scored_eval_v2/results.json');manifest=read(root/'scored_eval_v2/score_manifest.json')
        assert manifest['outputs']['results.json']==sha(root/'scored_eval_v2/results.json')
        assert manifest['authoritative'] and manifest['requested_contract_satisfied']
        assert manifest['metric_identity']['release_sha256']==catalog['metric_release_sha256']
        assert d['data_identity']['pack_id']==meta['pack_id']
        for path,pin in manifest['metric_identity']['source_sha256'].items():
            if path.startswith('jflux2/fasteval/eval_v2/'):
                assert sha(J/path)==pin;PINS[str(J/path)]=pin
        for name in ['samples.json','samples.npz']:
            p=L.PACKS/'bench_v2_val'/name;assert sha(p)==d['data_identity']['file_sha256'][name];PINS[str(p)]=sha(p)
        docs[model]=d
    canonical=docs['genlv'];canroot=dict(units)['genlv']
    recover=read(RUNS/'saliency_depth_diagnostic/recovery_plan.json')
    producer=read(Path(recover['source']));assert sha(recover['source'])==recover['source_sha256']
    units.append(('analogist',RUNS/'saliency_depth_diagnostic/analogist'))
    PINS[str(SITE/'scripts/depth_output_adapter.py')]=sha(SITE/'scripts/depth_output_adapter.py')
    rows=[];summary=[]
    for model,root in units:
        recipe=defaults[model]
        published=next(c for c in catalog['candidates'] if c['recipe']==recipe and c['task']=='relative_depth')
        for q in range(17):
            ref=canonical['episodes'][f'relative_depth@q{q}']
            target=image(canroot,ref['artifacts']['query_target'])
            if model=='analogist':
                e=next(e for e in producer['episodes'].values() if e['task']=='relative_depth' and e['q']==q)
                descs=e['artifacts']['candidates'];assert len(descs)==8 and not e['uses_query_target_for_prompt']
            else:
                e=docs[model]['episodes'][f'relative_depth@q{q}'];tag=next(iter(e['readouts']))
                descs=[e['artifacts']['readouts'][tag]['real']]
            assert e['query_record']==ref['query_record']
            for field in ['query_input','query_target']:
                assert e['artifacts'][field]['pixels_sha256']==ref['artifacts'][field]['pixels_sha256']
            ix=index[('relative_depth',e['query_record'])]
            demos=[index[('relative_depth',x)] for x in e['demo_records']]
            ctx=L.build_ctx('relative_depth','bench_v2_val',meta['records'][ix],ix,demos,target)
            dataset=ref['query_source'];lo,hi=P.DEPTH_RANGE_M[dataset]
            with np.errstate(divide='ignore'):depth=1/ctx['gt_inv']
            scored=ctx['gt_scored'] & P.depth_eval_region(dataset,target.shape) & (depth>=lo) & (depth<=hi)
            kwargs=dict(min_inv=P.depth_min_inv(dataset,ctx))
            for candidate,desc in enumerate(descs):
                pred=image(root,desc)
                decoded,invalid=decode_depth_prediction(pred,model=model) if model in ['painter','vict'] else PP.depth_grey(pred)
                before=M.depth_midas(decoded,invalid,ctx['gt_inv'],scored,**kwargs)
                after=depth_signed_scale(decoded,invalid,ctx['gt_inv'],scored,**kwargs)
                assert after==reference(decoded,invalid,ctx['gt_inv'],scored,**kwargs)
                assert before['coverage']==after['coverage']
                if after['scale']>=0:assert before==after
                else:assert before['scale']==0
                # A separate centered-covariance fit checks the signed solution.
                valid=scored & ~invalid;x=decoded[valid];y=ctx['gt_inv'][valid]
                if x.size>=2 and np.ptp(x)>0:
                    slope=float(np.sum((x-x.mean())*(y-y.mean()))/np.sum((x-x.mean())**2))
                    intercept=float(y.mean()-slope*x.mean())
                    assert np.isclose(slope,after['scale'],rtol=1e-8,atol=1e-8)
                    aligned=np.full(scored.sum(),kwargs['min_inv']);aligned[valid[scored]]=slope*x+intercept
                    aligned=np.maximum(aligned,kwargs['min_inv'])
                    ratio=np.maximum(aligned/ctx['gt_inv'][scored],ctx['gt_inv'][scored]/aligned)
                    assert float(np.mean(ratio<1.25))==after['delta1']
                    old_residual=np.mean((before['scale']*x+before['shift']-y)**2)
                    new_residual=np.mean((after['scale']*x+after['shift']-y)**2)
                    assert new_residual<=old_residual+1e-12
                rows.append(dict(model=model,recipe=recipe,q=q,query_record=e['query_record'],candidate=candidate,
                    prediction_sha256=desc['sha256'],before=before,after=after))
            current=[r for r in rows if r['model']==model and r['q']==q]
            expected=next(x for x in published['queries'] if x['q']==q)
            assert expected['query_record']==e['query_record']
            assert abs(float(np.mean([r['before']['delta1'] for r in current]))-expected['score'])<1e-10
        sub=[r for r in rows if r['model']==model]
        queries=[dict(q=q,query_record=next(r['query_record'] for r in sub if r['q']==q),
            before=float(np.mean([r['before']['delta1'] for r in sub if r['q']==q])),
            after=float(np.mean([r['after']['delta1'] for r in sub if r['q']==q]))) for q in range(17)]
        s=dict(model=model,recipe=recipe,queries=queries,candidate_images=len(sub),
            before=float(np.mean([q['before'] for q in queries])),after=float(np.mean([q['after'] for q in queries])),
            negative_scales=sum(r['after']['scale']<0 for r in sub),remaining_zero_scales=sum(r['after']['scale']==0 for r in sub),
            wins=sum(q['after']>q['before']+1e-12 for q in queries),losses=sum(q['after']<q['before']-1e-12 for q in queries),
            abs_rel_before=float(np.mean([r['before']['abs_rel'] for r in sub])),abs_rel_after=float(np.mean([r['after']['abs_rel'] for r in sub])))
        summary.append(s);print(json.dumps({k:v for k,v in s.items() if k!='queries'}),flush=True)
    write('input_sha256.json',PINS)
    write('results.json',dict(status='PASS',contract=CONTRACT,pack_id=meta['pack_id'],supervision_version=6,
        base_metric_release_sha256=catalog['metric_release_sha256'],summary=summary,per_candidate=rows,
        original_query_means_replayed=102,candidate_images=len(rows),
        independent_covariance_alignment_check=True,frozen_function_minus_one_branch_check=True,fixtures_pass=True,
        main_table_changed=False,inference_runs=0,inference_metadata_added=False,
        sign_selected_using_query_ground_truth=True,script_sha256=sha(__file__),adapter_sha256=sha(ROOT/'signed_depth_metric.py'),
        limitation='Sensitivity comparison for six locally recovered current recipes. Does not replace the published frozen-protocol depth scores. Analogist uses authenticated original producer images and all eight candidates; original scoring manifests remain unavailable.'))

if __name__=='__main__':main()
