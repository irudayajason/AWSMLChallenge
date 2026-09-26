import csv, tempfile, math, json, random, itertools
from pathlib import Path
import er
import numpy as np

er.metric_tests()
# Exhaustive small-state proof check of vectorized threshold search, including ties/singletons.
rng=random.Random(42)
rows=[]
for i in range(150):
    truth={j for j in range(6) if rng.random()<.35}
    scored=[(j,rng.choice([0,.1,.5,.9,1])) for j in range(8) if rng.random()<.65]
    rows.append((i,truth,scored))
_,_,grid,values=er.threshold_search(lambda:iter(rows),[0,.1,.2,.5,.9,1])
assert np.allclose(values,[np.mean([er.entity_f05(t,er.decide(s,g)) for q,t,s in rows]) for g in grid])
print('PASS metric, empty-universe accounting, tie-aware threshold sweep')
assert er.tokens('ग्लोबल')==['ग्लोबल']
assert er.make_views('<NULL>','address',{})['has'] is False
assert er.make_views('#hanumaneducational','name',{})['core']=='hanumaneducational'
for word in ['ग्लोबल','குளோபல்','Sun पावर Provision','Frères']:
    v=er.make_views(word,'name',{})
    assert v['raw']==word
    print('Transliteration:',word, '=>',v['roman'])

work=Path(tempfile.mkdtemp(prefix='er_plan_'))
def write(path,fields,rows):
    with open(path,'w',encoding='utf8',newline='') as f:
        w=csv.writer(f,delimiter='\t');w.writerow(fields);w.writerows(rows)

def dataset(root,split,n):
    root.mkdir();sources={1:[],2:[],3:[]};truth=[]
    countries=['US','India'] if split=='train' else ['France','India','US','']
    for i in range(n):
        country=countries[i%len(countries)]
        base=f'Business {i} Alpha Services'
        sources[1].append([f'S1-{i}',base,f'{i+1} Orchard Road',country])
        sources[2].append([f'S2-{i}',base, '<NULL>' if i%7==0 else f'{i+1} Orchard Rd',country])
        sources[3].append([f'S3-{i}',base+(' LLC LLC' if i%3==0 else ''),f'{i+1} Orchard Road',country])
        # some true singletons, whose similar-looking candidates must be rejected
        matches=[] if i%11==0 else [f'S2-{i}',f'S3-{i}']
        truth.append([f'S1-{i}',','.join(matches)])
    for source,rows in sources.items():write(root/f'{split}_source{source}.tsv',er.REQUIRED,rows)
    if split=='train':write(root/'train_ground_truth.tsv',['source1_entity_id','matched_entity_ids'],truth)

dataset(work/'train','train',260);dataset(work/'test','test',12)
db=er.connect(work/'component.sqlite')
for source in (1,2,3):er.ingest(db,work/'train'/f'train_source{source}.tsv',source)
er.load_truth(db,work/'train'/'train_ground_truth.tsv')
q1=er.lookup_id(db,'S1-1',{1});q2=er.lookup_id(db,'S1-2',{1});t=er.lookup_id(db,'S2-1',{2})
db.execute('INSERT INTO truth VALUES(?,?)',(q2,t));db.commit()
er.assign_splits(db,work/'groups')
assert db.execute('SELECT count(DISTINCT split) FROM gt_entities WHERE q IN (?,?)',(q1,q2)).fetchone()[0]==1
assert er.pair_groups(db,np.array([q1,q2])).tolist()[0]==er.pair_groups(db,np.array([q1,q2])).tolist()[1]
er.configure_feature_weights(db,work/'feature_idf.npy')
metas=er.build_all_indexes(db,work/'small_indexes',('char','rare','translit','address'),shard=31)
q=er.fetch_record(db,q1)
hits=er.retrieve_batch(db,metas,[q],k=60,country_verified=True)
assert t in {c['t'] for c in hits[q1]}
assert hits==er.retrieve_batch(db,metas,[q],k=60,country_verified=True)
assert er.route_countries('france',{'france','india',''},True)==['','france']
assert er.route_countries('',{'france','india',''},True)==['','france','india']
print('PASS disk IDs, connected-component split, sparse sharding, deterministic fusion, routing')
# Optional MinHash name-only index: catalog address must not dilute similarity.
record=er.fetch_record(db,er.lookup_id(db,'S2-7',{2}))
query=er.fetch_record(db,er.lookup_id(db,'S1-7',{1}))
lsh=er.BoundedMinHash('name',max_records=10)
lsh.add(record)
assert record['rid'] in {h.t for h in lsh.query(db,query)}
assert not hasattr(lsh,'s2s3_minhashes')
mmeta=er.build_all_indexes(db,work/'minhash_indexes',('minhash_name','minhash_joint'),shard=31)
merged=er.retrieve_batch(db,mmeta,[query],k=40,country_verified=True)
assert record['rid'] in {c['t'] for c in merged[query['rid']]}
print('PASS optional name-only MinHash and serialized name/joint shard integration')
# Every output S1 must appear, including no candidates.
qids=[r[0] for r in db.execute('SELECT rid FROM records WHERE source=1 ORDER BY rid')]
er.write_outputs(db,qids,lambda q:[],.5,work/'empty_output')
er.validate_outputs(db,work/'empty_output'/'matching_results.tsv',work/'empty_output'/'candidate_pairs.tsv')
print('PASS complete output coverage and subset constraints')
db.close()
# Verify the supervised token correspondence learner, on FIT labels only.
spell_dir=work/'spell';spell_dir.mkdir()
write(spell_dir/'s1.tsv',er.REQUIRED,[[f'S1-{i}',f'Global {i} Retail','1 Main Road','India'] for i in range(80)])
write(spell_dir/'s2.tsv',er.REQUIRED,[[f'S2-{i}',f'குளோபல் {i} Retail','1 Main Road','India'] for i in range(80)])
write(spell_dir/'gt.tsv',['source1_entity_id','matched_entity_ids'],[[f'S1-{i}',f'S2-{i}'] for i in range(80)])
sdb=er.connect(spell_dir/'records.sqlite')
er.ingest(sdb,spell_dir/'s1.tsv',1);er.ingest(sdb,spell_dir/'s2.tsv',2)
er.load_truth(sdb,spell_dir/'gt.tsv');er.assign_splits(sdb,spell_dir/'groups')
rules,stats=er.learn_spelling(sdb)
assert rules.get('kulopal')=='global', (rules,stats)
er.apply_spelling(sdb,rules)
assert er.fetch_record(sdb,er.lookup_id(sdb,'S2-0',{2}))['name']['corrected']=='global 0 retail'
sdb.close()
print('PASS FIT-only learned Tamil-to-English spelling correspondence')
report=er.core_run(work/'train',work/'test',work/'run',work/'output',kinds=('char','rare','translit'),k=40)
print('PASS core_run ingest -> train -> calibrate -> threshold -> France test -> validated TSV')
print(json.dumps(report,indent=2))
print('TEST_WORKSPACE',work)
