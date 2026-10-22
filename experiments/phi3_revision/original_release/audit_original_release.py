"""Audit public release identity, strict labels and compact-corpus correspondence."""
from pathlib import Path
import collections,hashlib,json,sys,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core import canonical_hash,parse_instance,parse_matrix,validate_schedule
ROOT=Path(__file__).resolve().parent
summary={'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'sources':{}}
ids={}
for name in ['test_2000.json','jssp_llm_format_120k.json']:
 print('Loading',name,flush=True)
 rows=json.loads((ROOT/name).read_text()); counts=collections.Counter(); errors=collections.Counter(); first=[]; unique=set(); collisions=0; signatures=[]; dimensions=collections.Counter(); lo=float('inf');hi=0
 for i,row in enumerate(rows):
  try:
   inst=parse_instance(row['prompt_machines_first']); instance_id=canonical_hash(inst)
   if 'matrix' in row and instance_id!=canonical_hash(parse_matrix(row['matrix'])): raise ValueError('Prompt-matrix identity mismatch')
   if instance_id in unique: collisions+=1
   unique.add(instance_id); signatures.append(instance_id); dimensions[f'{inst.n_jobs}x{inst.n_machines}']+=1
   for op in inst.operations: lo=min(lo,op.duration);hi=max(hi,op.duration)
   result=validate_schedule(inst,row['output'])
   if result.feasible: counts['labels_feasible']+=1
   else:
    counts['labels_infeasible']+=1;errors.update(result.errors)
    if len(first)<20:first.append({'index':i,'errors':list(result.errors),'details':list(result.details)[:8]})
  except Exception as e:
   counts['parse_errors']+=1
   if len(first)<20:first.append({'index':i,'exception':str(e)})
  if i%10000==0: print(name,i,'of',len(rows),dict(counts),flush=True)
 ids[name]=unique
 signature_hash=hashlib.sha256('\n'.join(signatures).encode()).hexdigest()
 summary['sources'][name]={'count':len(rows),'columns':list(rows[0]),'counts':dict(counts),'errors':dict(errors),'examples':first,'unique_instances':len(unique),'duplicate_instances':collisions,'dimensions':dict(sorted(dimensions.items())),'minimum_duration':lo,'maximum_duration':hi,'ordered_instance_signatures_sha256':signature_hash}
 (ROOT/(name+'.instance_hashes.json')).write_text(json.dumps(signatures)+'\n')
 del rows
 (ROOT/'audit.json').write_text(json.dumps(summary,indent=2)+'\n')
summary['train_test_overlap']=len(ids['test_2000.json']&ids['jssp_llm_format_120k.json'])
summary['completed_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
(ROOT/'audit.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2),flush=True)
