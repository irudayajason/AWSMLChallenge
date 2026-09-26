# SECTION 1
from __future__ import annotations
import csv, hashlib, heapq, json, math, os, re, sqlite3, time, unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize

CHANNELS = {'exact': 0, 'char': 1, 'rare': 2, 'translit': 3,
            'address': 4, 'minhash_name': 5, 'minhash_joint': 6}
@dataclass(frozen=True)
class Hit:
    q: int
    t: int
    source: int
    channel: str
    score: float

# SECTION 2
LIMITS = dict(chunk=5000, shard=100000, query_batch=128,
              feature_batch=8192, text_chars=4096, hash_dim=2**18,
              sqlite_cache_mib=128, max_train_pairs=1500000)
def dependency_manifest():
    from importlib.metadata import version
    packages = ['numpy', 'scipy', 'scikit-learn', 'sparse_dot_topn',
                'pyicu-wheels', 'rapidfuzz', 'lightgbm']
    return {p: version(p) for p in packages}

def memory_estimates(nrows, nnz_per_row, queries, k, features, pairs,
                     dimensions=2**18, index_bytes=4):
    nnz = nrows * nnz_per_row
    return {'csr_bytes': nnz*(4+index_bytes)+(nrows+1)*index_bytes,
            'topk_output_bytes': queries*k*(4+index_bytes)+(queries+1)*index_bytes,
            'feature_bytes': pairs*features*4,
            'idf_bytes': dimensions*4, 'df_bytes': dimensions*8}

# SECTION 3
REQUIRED = ['entity_id', 'business_name', 'business_address', 'country']
NULL_ANGLE = re.compile(r'<\s*(?:null|nan|none)\s*>', re.I)
NULL_WORD = re.compile(r'\b(?:null|nan|none)\b', re.I)
def missing_clean(value):
    text = '' if value is None else str(value)
    text = NULL_ANGLE.sub(' ', text)       # MUST precede generic removal
    text = NULL_WORD.sub(' ', text)
    return ' '.join(text.split()).strip(' ,;|')

def read_tsv(path, required=REQUIRED, chunk=5000):
    csv.field_size_limit(65536)
    with open(path, encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f, delimiter='\t')
        if reader.fieldnames != required:
            raise ValueError(f'{path}: expected columns {required}; got {reader.fieldnames}')
        batch = []
        for line, row in enumerate(reader, 2):
            if None in row or any(v is None for v in row.values()):
                raise ValueError(f'{path}:{line}: malformed row')
            if required == REQUIRED:
                if any(len(row[k]) > LIMITS['text_chars'] for k in REQUIRED[1:3]):
                    raise ValueError(f'{path}:{line}: field exceeds profiled safety limit')
                if not row['entity_id'] or len(row['entity_id']) > 128:
                    raise ValueError('Invalid entity ID')
            batch.append(row)
            if len(batch) == chunk:
                yield batch
                batch = []
        if batch:
            yield batch

def connect(path):
    db = sqlite3.connect(path)
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('PRAGMA temp_store=FILE')
    db.execute('PRAGMA cache_size=-131072')
    db.executescript('''
      CREATE TABLE IF NOT EXISTS records(
        rid INTEGER PRIMARY KEY, eid TEXT UNIQUE NOT NULL,
        source INTEGER NOT NULL CHECK(source IN (1,2,3)),
        country TEXT NOT NULL, name_key TEXT NOT NULL,
        missing INTEGER NOT NULL, payload TEXT NOT NULL);
      CREATE INDEX IF NOT EXISTS rec_exact ON records(country,source,name_key,rid);
      CREATE INDEX IF NOT EXISTS rec_route ON records(country,source,rid);
      CREATE TABLE IF NOT EXISTS gt_entities(q INTEGER PRIMARY KEY, split TEXT);
      CREATE TABLE IF NOT EXISTS truth(q INTEGER NOT NULL,t INTEGER NOT NULL,
                                      PRIMARY KEY(q,t));
      CREATE INDEX IF NOT EXISTS truth_target ON truth(t,q);
    ''')
    return db

def ingest(db, path, source, spell=None, workers=1):
    from concurrent.futures import ProcessPoolExecutor
    pool = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        for batch in read_tsv(path, chunk=LIMITS['chunk']):
            args = [(r, source, spell or {}) for r in batch]
            out = pool.map(normalize_job, args, chunksize=128) if pool else map(normalize_job,args)
            with db:
                db.executemany('INSERT INTO records(eid,source,country,name_key,missing,payload) '
                               'VALUES(?,?,?,?,?,?)', out)
    finally:
        if pool:
            pool.shutdown()

def fetch_many(db,ids):
    ids=sorted(set(map(int,ids)));out={}
    for start in range(0,len(ids),800):
        subset=ids[start:start+800]
        sql='SELECT rid,payload,source,country FROM records WHERE rid IN ('+','.join('?' for _ in subset)+')'
        for rid,payload,source,country in db.execute(sql,subset):
            rec=json.loads(payload);rec.update(rid=rid,source=source,country=country)
            out[rid]=rec
    if len(out)!=len(ids):raise ValueError('Missing target records')
    return out

def fetch_record(db, rid):
    row = db.execute('SELECT payload,source,country FROM records WHERE rid=?',(int(rid),)).fetchone()
    if row is None:
        raise KeyError(rid)
    rec = json.loads(row[0]); rec.update(rid=int(rid),source=row[1],country=row[2])
    return rec

def lookup_id(db, eid, allowed):
    row = db.execute('SELECT rid,source FROM records WHERE eid=?',(eid,)).fetchone()
    if row is None or row[1] not in allowed:
        raise ValueError(f'Unknown or wrong-source ID: {eid}')
    return int(row[0])

def load_truth(db, path):
    fields = ['source1_entity_id','matched_entity_ids']
    for batch in read_tsv(path, fields):
        with db:
            for row in batch:
                q = lookup_id(db,row[fields[0]],{1})
                db.execute('INSERT INTO gt_entities(q) VALUES(?)',(q,))
                ids = row[fields[1]].split(',') if row[fields[1]] else []
                if len(ids) != len(set(ids)):
                    raise ValueError('Duplicate ground-truth ID')
                db.executemany('INSERT INTO truth VALUES(?,?)',
                               [(q,lookup_id(db,t,{2,3})) for t in ids])
    total = db.execute('SELECT count(*) FROM records WHERE source=1').fetchone()[0]
    if total != db.execute('SELECT count(*) FROM gt_entities').fetchone()[0]:
        raise ValueError('Ground truth must cover EVERY training S1')

def truth_for(db,q):
    return {int(r[0]) for r in db.execute('SELECT t FROM truth WHERE q=?',(int(q),))}

# SECTION 4
def unicode_view(text):
    return unicodedata.normalize('NFKC', text)
def tokens(text):
    # Keep Indic combining marks attached to letters; Python \w alone loses them.
    parts=[];word=[]
    for c in text.casefold():
        if c.isalnum() or (unicodedata.category(c).startswith('M') and word):
            word.append(c)
        elif word:
            parts.append(''.join(word));word=[]
    if word:parts.append(''.join(word))
    return parts
def conservative_view(text):
    return ' '.join(tokens(unicode_view(text)))
def fold_accents(text):
    # Preserve non-Latin marks; Latin combining accents are folded only.
    out=[]; latin=False
    for c in unicodedata.normalize('NFD',text):
        if not unicodedata.combining(c):
            latin = 'LATIN' in unicodedata.name(c,'')
            out.append(c)
        elif not latin:
            out.append(c)
    return unicodedata.normalize('NFC',''.join(out))

def make_views(raw, field, spell):
    uni=unicode_view(raw)
    cleaned=clean_name(uni) if field=='name' else clean_address(uni)
    cons=conservative_view(cleaned)
    translit=conservative_view(transliterate(cleaned))
    folded=fold_accents(cons)
    roman=fold_accents(translit)
    corrected=' '.join(spell.get(t,t) if field=='name' else t for t in tokens(roman))
    core,legal=core_and_legal(corrected)
    return dict(raw=raw,unicode=uni,conservative=cons,transliterated=translit,
                accent_folded=folded,roman=roman,corrected=corrected,
                core=core if field=='name' else corrected,
                legal=legal if field=='name' else '',
                has=bool(tokens(cons)),script=script_profile(uni))

def normalize_job(args):
    row,source,spell=args
    if not row['entity_id'].startswith(f'S{source}-'):
        raise ValueError('Source prefix does not match input file')
    name=make_views(row['business_name'],'name',spell)
    addr=make_views(row['business_address'],'address',spell)
    country=conservative_view(missing_clean(row['country']))
    payload=json.dumps(dict(name=name,address=addr),ensure_ascii=False,separators=(',',':'))
    return row['entity_id'],source,country,name['core'],int(not addr['has']),payload

# SECTION 5
SCRIPT_RANGES=((0x0900,0x097F),(0x0980,0x09FF),(0x0A00,0x0A7F),
               (0x0A80,0x0AFF),(0x0B00,0x0B7F),(0x0B80,0x0BFF),
               (0x0C00,0x0C7F),(0x0C80,0x0CFF),(0x0D00,0x0D7F))
_ICU=None
def script_profile(text):
    letters=[c for c in text if c.isalpha()]
    if not letters: return {'indic':0.0,'latin':0.0,'other':0.0}
    indic=sum(any(lo<=ord(c)<=hi for lo,hi in SCRIPT_RANGES) for c in letters)
    latin=sum('LATIN' in unicodedata.name(c,'') for c in letters)
    n=len(letters)
    return dict(indic=indic/n,latin=latin/n,other=(n-indic-latin)/n)

def transliterate(text):
    global _ICU
    if text.isascii(): return text
    if not any(any(lo<=ord(c)<=hi for lo,hi in SCRIPT_RANGES) for c in text):
        return text
    if _ICU is None:
        import icu
        _ICU=icu.Transliterator.createInstance('Any-Latin')
    return _ICU.transliterate(text)

def learn_spelling(db,limit=100000,min_support=20,purity=.97,max_rules=5000):
    # Training labels ONLY. Unique one-to-one residual token alignment is conservative.
    from rapidfuzz.fuzz import ratio
    counts=Counter(); inspected=0
    for q,t in db.execute("SELECT t.q,t.t FROM truth t JOIN gt_entities g ON g.q=t.q "
                          "WHERE g.split='fit' ORDER BY t.q,t.t LIMIT ?",(limit,)):
        a,b=fetch_record(db,q),fetch_record(db,t)
        if b['name']['script']['indic']==0: continue
        left=set(tokens(b['name']['roman'])); right=set(tokens(a['name']['roman']))
        common=left&right; left-=common;right-=common
        if len(left)==len(right)==1:
            x,y=next(iter(left)),next(iter(right))
            if min(len(x),len(y))>=4 and ratio(x,y)>=45:
                counts[x,y]+=1
        inspected+=1
    totals=Counter()
    for (x,y),n in counts.items(): totals[x]+=n
    best={}
    for (x,y),n in sorted(counts.items(),key=lambda z:(-z[1],z[0])):
        if x not in best and n>=min_support and n/totals[x]>=purity:
            best[x]=y
            if len(best)>=max_rules: break
    return best,dict(inspected=inspected,aligned_pairs=sum(counts.values()))

def apply_spelling(db,rules,chunk=5000):
    last=0
    while True:
        rows=db.execute('SELECT rid,payload FROM records WHERE rid>? ORDER BY rid LIMIT ?',
                        (last,chunk)).fetchall()
        if not rows: break
        updates=[]
        for rid,payload in rows:
            rec=json.loads(payload)
            for field in ['name','address']:
                v=rec[field]
                v['corrected']=' '.join(rules.get(t,t) if field=='name' else t for t in tokens(v['roman']))
                v['core'],v['legal']=core_and_legal(v['corrected']) if field=='name' else (v['corrected'],'')
            updates.append((rec['name']['core'],json.dumps(rec,ensure_ascii=False),rid))
        with db: db.executemany('UPDATE records SET name_key=?,payload=? WHERE rid=?',updates)
        last=rows[-1][0]

# SECTION 6
NOISE_PATTERNS={
 'leading_dash': re.compile(r'^\s*--+'),
 'leading_star': re.compile(r'^\s*\*{2,}'),
 'leading_angle': re.compile(r'^\s*<<+'),
 'address_hash': re.compile(r'^\s*##+'),
 'url': re.compile(r'(?:https?://|www\.)\S+',re.I),
 'pipe': re.compile(r'\|'),
 'hashtag': re.compile(r'#(?=\w)'),
 'llc_repeat':re.compile(r'\bLLC(?:\s+LLC)+\b',re.I),
 'pvt_repeat':re.compile(r'\bPvt(?:\s+Pvt)+\b',re.I),
 'ltd_repeat':re.compile(r'\bLtd(?:\s+Ltd)+\b',re.I),
 'private_repeat':re.compile(r'\bPrivate(?:\s+Private)+\b',re.I),
}
STUTTER=re.compile(r'\b(LLC|Pvt|Ltd|Private|Inc|Corp|Limited|Corporation|Company)(?:\s+\1)+\b',re.I)
def clean_name(text):
    text=missing_clean(text)
    text=re.sub(r'^\s*[-*<>#~]+\s*','',text)
    text=NOISE_PATTERNS['url'].sub(' ',text)
    text=NOISE_PATTERNS['hashtag'].sub('',text) # keep name/numeric content
    text=NOISE_PATTERNS['pipe'].sub(' ',text)
    text=STUTTER.sub(r'\1',text)
    text=re.sub(r'\s*&\s*',' and ',text)
    return ' '.join(text.split())

def count_noise(path):
    counts=Counter()
    for batch in read_tsv(path):
        for row in batch:
            for key,regex in NOISE_PATTERNS.items():
                field='business_address' if key=='address_hash' else 'business_name'
                counts[key]+=bool(regex.search(row[field]))
            a=row['business_address']
            counts['blank_address']+=not a.strip()
            counts['angle_null']+=bool(NULL_ANGLE.search(a))
            counts['generic_null']+=bool(NULL_WORD.search(a))
            counts['unusable_after_clean']+=not bool(tokens(clean_address(a)))
    return dict(counts)

# SECTION 7
# Initial forms supported by the supplied TRAIN distribution; validate mapping gain.
LEGAL_MAP={'incorporated':'inc','inc':'inc','corporation':'corp','corp':'corp',
           'pvt':'private','private':'private','ltd':'limited','limited':'limited',
           'llc':'llc','co':'company','company':'company'}
def core_and_legal(text):
    seq=tokens(text); suffix=[]
    while seq and seq[-1] in LEGAL_MAP:
        term=LEGAL_MAP[seq.pop()]
        if not suffix or term!=suffix[-1]: suffix.append(term)
    # Keep full name when every token is a legal word; never create accidental empty names.
    return (' '.join(seq) if seq else ' '.join(tokens(text)), ' '.join(reversed(suffix)))

def suffix_counts(db):
    counts=Counter()
    for (payload,) in db.execute('SELECT payload FROM records WHERE source=1'):
        seq=tokens(json.loads(payload)['name']['conservative'])
        if seq and seq[-1] in LEGAL_MAP: counts[seq[-1]]+=1
    return dict(counts)

# SECTION 8
ADDR_RULES=[(r'\bb\s*/\s*h\b','behind'),(r'\bopp\b\.?','opposite'),
            (r'\bnr\b\.?','near'),(r'\badj\b\.?','adjacent'),
            (r'\brd\b\.?','road'),(r'\bave\b\.?','avenue'),
            (r'\bblvd\b\.?','boulevard'),(r'\bln\b\.?','lane'),
            (r'\bhwy\b\.?','highway'),(r'\bste\b\.?','suite')]
ADDR_RULES=[(re.compile(p,re.I),v) for p,v in ADDR_RULES]
def clean_address(text):
    text=missing_clean(text)
    text=NOISE_PATTERNS['address_hash'].sub('',text)
    text=NOISE_PATTERNS['url'].sub(' ',text)
    for regex,replacement in ADDR_RULES: text=regex.sub(replacement,text)
    return ' '.join(text.split()).strip(' ,;|')

def address_parts(text):
    seq=tokens(text)
    numbers=set(re.findall(r'\b\d+[a-z]?\b',text.casefold()))
    landmarks=set(seq)&{'near','opposite','behind','adjacent'}
    return set(seq),numbers,landmarks

# SECTION 9
# Sparse hashed TF-IDF: bounded vocabulary; collisions must be measured in ablations.
def vectorizer(kind,dim=2**18):
    if kind in {'char','translit'}:
        return HashingVectorizer(n_features=dim,analyzer='char_wb',ngram_range=(3,5),
                                 alternate_sign=False,norm=None,dtype=np.float32)
    return HashingVectorizer(n_features=dim,analyzer='word',tokenizer=tokens,
                             token_pattern=None,lowercase=False,alternate_sign=False,
                             norm=None,dtype=np.float32)

def field_text(rec,kind):
    if kind=='char': return rec['name']['accent_folded']
    if kind=='translit': return rec['name']['corrected']
    if kind=='rare': return rec['name']['core']
    if kind=='address': return rec['address']['corrected']
    raise ValueError(kind)

def csr_bytes(x):
    return x.data.nbytes+x.indices.nbytes+x.indptr.nbytes

def make_index(db,root,country,source,kind,dim=2**18,shard=100000,max_df=.20):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    vec=vectorizer(kind,dim);df=np.zeros(dim,dtype=np.int64);n=0;parts=[]
    def chunks():
        cur=db.execute('SELECT rid,payload FROM records WHERE source=? AND country=? ORDER BY rid',
                       (source,country))
        ids=[];texts=[];estimate=0
        for rid,payload in cur:
            text=field_text(json.loads(payload),kind)
            # Upper bound on emitted char_wb grams, or word tokens, BEFORE vectorization.
            cost=3*(len(text)+2*len(text.split())+1) if kind in {'char','translit'} else len(text)+1
            if ids and (len(ids)>=shard or estimate+cost>8000000):
                yield np.asarray(ids,np.int64),texts
                ids=[];texts=[];estimate=0
            ids.append(rid);texts.append(text);estimate+=cost
        if ids:yield np.asarray(ids,np.int64),texts
    for ids,texts in chunks():
        x=vec.transform(texts).tocsr();x.sum_duplicates();x.sort_indices()
        df+=np.bincount(x.indices,minlength=dim)
        part=root/f'part-{len(parts):05d}'
        sparse.save_npz(str(part)+'.npz',x);np.save(str(part)+'.ids.npy',ids)
        parts.append(str(part));n+=len(ids)
    idf=(np.log((1+n)/(1+df))+1).astype(np.float32)
    # Rare-word channels exclude frequent postings; never prune all evidence elsewhere.
    if kind in {'rare','address'}:
        idf[df>max_df*max(n,1)]=0
    np.save(root/'idf.npy',idf)
    for part in parts:
        x=sparse.load_npz(part+'.npz').tocsr()
        np.log1p(x.data,out=x.data);x.data*=idf[x.indices]
        x.eliminate_zeros();normalize(x,norm='l2',copy=False)
        xt=x.T.tocsr();xt.sort_indices()
        for key,array in [('data',xt.data),('indices',xt.indices),('indptr',xt.indptr)]:
            np.save(part+'.t.'+key+'.npy',array)
        Path(part+'.shape.json').write_text(json.dumps(xt.shape))
        Path(part+'.npz').unlink()
    meta=dict(country=country,source=source,kind=kind,dim=dim,parts=parts,n=n,
              root=str(root),max_df=max_df)
    (root/'meta.json').write_text(json.dumps(meta))
    return meta

def route_countries(query_country,available,country_verified):
    if not country_verified or not query_country:
        return sorted(available)
    return sorted(c for c in available if c in {query_country,''})

def query_index(meta,query_records,k=20,threads=1):
    from sparse_dot_topn import sp_matmul_topn
    if k<=0: return []
    vec=vectorizer(meta['kind'],meta['dim'])
    idf=np.load(Path(meta['root'])/'idf.npy',mmap_mode='r')
    qx=vec.transform([field_text(r,meta['kind']) for r in query_records]).tocsr()
    np.log1p(qx.data,out=qx.data);qx.data*=idf[qx.indices]
    qx.eliminate_zeros();normalize(qx,norm='l2',copy=False)
    best=[{} for _ in query_records]
    for part in meta['parts']:
        arrays=[np.load(part+'.t.'+key+'.npy',mmap_mode='c') for key in ['data','indices','indptr']]
        xt=sparse.csr_matrix(tuple(arrays),shape=tuple(json.loads(Path(part+'.shape.json').read_text())),copy=False)
        ids=np.load(part+'.ids.npy',mmap_mode='r')
        if not len(ids): continue
        result=sp_matmul_topn(qx,xt,top_n=min(k,len(ids)),
                             threshold=0.0,sort=True,n_threads=threads)
        for i in range(len(query_records)):
            lo,hi=result.indptr[i:i+2]
            d=best[i]
            for j,s in zip(result.indices[lo:hi],result.data[lo:hi]):
                if s>0: d[int(ids[j])]=max(d.get(int(ids[j]),0.0),float(s))
            best[i]=dict(sorted(d.items(),key=lambda z:(-z[1],z[0]))[:k])
        del xt,result,arrays
    return [Hit(int(r['rid']),t,meta['source'],meta['kind'],s)
            for r,d in zip(query_records,best) for t,s in d.items()]

def exact_hits(db,q,countries,per_source=20):
    key=q['name']['core']
    if not key:return [],False
    hits=[];overflow=False
    for country in countries:
        for source in (2,3):
            rows=db.execute('SELECT rid FROM records WHERE country=? AND source=? AND name_key=? '
                            'ORDER BY rid LIMIT ?', (country,source,key,per_source+1)).fetchall()
            overflow|=len(rows)>per_source
            hits.extend(Hit(q['rid'],int(r[0]),source,'exact',1.0) for r in rows[:per_source])
    return hits,overflow

class BoundedMinHash:
    # Optional, OFF by default. Exact bucket occupancy guards bound lsh.query allocation.
    def __init__(self,mode='name',max_records=50000,max_bucket=256,k=20,num_perm=64):
        from datasketch import MinHashLSH
        if mode not in {'name','joint'}:raise ValueError(mode)
        self.mode=mode;self.max_records=max_records;self.max_bucket=max_bucket;self.k=k
        self.num_perm=num_perm;self.lsh=MinHashLSH(threshold=.4,num_perm=num_perm)
        self.n=0;self.overflow=False
    def sketch(self,rec):
        from datasketch import MinHash
        ts=set(tokens(rec['name']['core']))
        if self.mode=='joint':ts|=set(tokens(rec['address']['corrected']))
        if not ts:return None
        mh=MinHash(num_perm=self.num_perm,seed=42)
        mh.update_batch([t.encode('utf8') for t in sorted(ts)])
        return mh
    def add(self,rec):
        if self.mode=='name' and rec['address']['has']:return
        if self.mode=='joint' and not rec['address']['has']:return
        if self.n>=self.max_records:raise MemoryError('Split MinHash shard before inserting')
        mh=self.sketch(rec)
        if mh is None:return
        # Pinned datasketch API: integration test these internals before adoption.
        keys=[self.lsh._H(mh.hashvalues[a:b]) for a,b in self.lsh.hashranges]
        if any(len(tab.get(key))>=self.max_bucket for tab,key in zip(self.lsh.hashtables,keys)):
            self.overflow=True
            return  # logged recall tradeoff; never silently claim complete coverage
        self.lsh.insert(int(rec['rid']),mh);self.n+=1
        # mh released here; NO persistent sketch dictionary
    def query(self,db,q):
        if self.mode=='joint' and not q['address']['has']:return []
        mh=self.sketch(q)
        if mh is None:return []
        ids=self.lsh.query(mh) # <= number of bands * max_bucket
        qt=set(tokens(q['name']['core']))
        if self.mode=='joint':qt|=set(tokens(q['address']['corrected']))
        hits=[]
        for t in ids:
            r=fetch_record(db,t);tt=set(tokens(r['name']['core']))
            if self.mode=='joint':tt|=set(tokens(r['address']['corrected']))
            score=len(qt&tt)/len(qt|tt) if qt|tt else 0.0
            hits.append(Hit(q['rid'],int(t),r['source'],'minhash_'+self.mode,score))
        return sorted(hits,key=lambda h:(-h.score,h.t))[:self.k]

def build_minhash_index(db,root,country,source,mode='name',shard=50000):
    # Optional index; build once, serialize bounded shards, then release each one.
    import pickle
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    missing=1 if mode=='name' else 0
    cur=db.execute('SELECT rid FROM records WHERE country=? AND source=? AND missing=? ORDER BY rid',
                   (country,source,missing))
    paths=[];overflows=0
    while True:
        ids=cur.fetchmany(shard)
        if not ids:break
        index=BoundedMinHash(mode,max_records=shard,k=BUDGETS['minhash_'+mode])
        for (rid,) in ids:index.add(fetch_record(db,rid))
        overflows+=int(index.overflow)
        path=root/f'minhash-{len(paths):05d}.pkl'
        with open(path,'wb') as f:pickle.dump(index,f)
        paths.append(str(path));del index
    return dict(kind='minhash_'+mode,country=country,source=source,parts=paths,
                overflow_shards=overflows)

def query_minhash_index(db,meta,queries,k=10):
    import pickle
    best={q['rid']:{} for q in queries}
    for path in meta['parts']:
        # Load only trusted, locally generated artifacts, never arbitrary pickle files.
        with open(path,'rb') as f:index=pickle.load(f)
        for q in queries:
            for h in index.query(db,q):
                d=best[q['rid']];old=d.get(h.t)
                if old is None or h.score>old.score:d[h.t]=h
            best[q['rid']]=dict((h.t,h) for h in sorted(best[q['rid']].values(),key=lambda h:(-h.score,h.t))[:k])
        del index
    return [h for d in best.values() for h in d.values()]

# SECTION 10
BUDGETS={'exact':20,'char':20,'rare':10,'translit':15,'address':10,
         'minhash_name':10,'minhash_joint':10}
def fuse(hits,k=60,coverage=2):
    if not hits:return []
    q=hits[0].q
    if any(type(h.q)!=int or type(h.t)!=int or h.q!=q for h in hits):
        raise TypeError('All channels must use ONE integer query ID and integer target IDs')
    groups=defaultdict(dict)
    for h in hits:
        key=h.channel,h.source
        old=groups[key].get(h.t)
        if old is None or h.score>old.score:groups[key][h.t]=h
    ranked={g:sorted(d.values(),key=lambda h:(-h.score,h.t))[:BUDGETS[g[0]]]
            for g,d in groups.items()}
    coverage=min(coverage,k//max(1,len(ranked)))
    by_t={};rank_score=defaultdict(float);mask=defaultdict(int);reserve=set()
    for g in sorted(ranked):
        for rank,h in enumerate(ranked[g],1):
            by_t[h.t]=h;rank_score[h.t]+=1/(60+rank);mask[h.t]|=1<<CHANNELS[h.channel]
            if rank<=coverage:reserve.add(h.t)
    if len(reserve)>k:raise ValueError('K too small for requested channel/source coverage')
    ordered=sorted(by_t,key=lambda t:(-rank_score[t],-bin(mask[t]).count("1"),t))
    chosen=reserve|set([t for t in ordered if t not in reserve][:k-len(reserve)])
    return [dict(q=q,t=t,source=by_t[t].source,mask=mask[t],rrf=rank_score[t])
            for t in ordered if t in chosen]

# SECTION 11
FEATURE_NAMES=['name_exact','core_exact','name_ratio','name_token_set','translit_ratio',
 'name_jaccard','rare_overlap','legal_equal','addr_ratio','addr_jaccard','numeric_overlap',
 'numeric_conflict','landmark_overlap','addr_missing','name_length_ratio','country_equal',
 'cross_script','indic_fraction','source3','channels','retrieved_exact','retrieved_char',
 'retrieved_rare','retrieved_translit','retrieved_address','retrieved_minhash_name',
 'retrieved_minhash_joint','rrf']
def jac(a,b):return len(a&b)/len(a|b) if a|b else 0.0

FEATURE_WORD_IDF=None
FEATURE_WORD_VEC=None

def configure_feature_weights(db,path):
    global FEATURE_WORD_IDF,FEATURE_WORD_VEC
    FEATURE_WORD_VEC=vectorizer('rare')
    if Path(path).exists():
        FEATURE_WORD_IDF=np.load(path,mmap_mode='r');return
    df=np.zeros(LIMITS['hash_dim'],np.int64);n=0
    cur=db.execute('SELECT payload FROM records WHERE source=1 ORDER BY rid')
    while True:
        rows=cur.fetchmany(5000)
        if not rows:break
        x=FEATURE_WORD_VEC.transform([json.loads(r[0])['name']['core'] for r in rows]).tocsr()
        x.sum_duplicates();df+=np.bincount(x.indices,minlength=len(df));n+=len(rows)
    FEATURE_WORD_IDF=(np.log((1+n)/(1+df))+1).astype(np.float32)
    np.save(path,FEATURE_WORD_IDF)

def pair_features(q,t,candidate,word_vec=None,word_idf=None):
    from rapidfuzz import fuzz
    if word_vec is None:
        word_vec,word_idf=FEATURE_WORD_VEC,FEATURE_WORD_IDF
    if word_vec is None or word_idf is None:
        raise ValueError('Call configure_feature_weights before building pair features')
    a,b=q['name'],t['name'];x,y=q['address'],t['address']
    if '_features' not in q:q['_features']=(set(tokens(a['core'])),address_parts(x['corrected']))
    if '_features' not in t:t['_features']=(set(tokens(b['core'])),address_parts(y['corrected']))
    at,(ax,an,al)=q['_features'];bt,(bx,bn,bl)=t['_features']
    overlap=jac(at,bt)
    if word_vec is not None:
        def weight(ts):
            if not ts:return 0.0
            v=word_vec.transform([' '.join(sorted(ts))])
            return float(word_idf[v.indices].sum())
        overlap=weight(at&bt)/max(weight(at|bt),1e-8)
    missing=not x['has'] or not y['has']
    cross=(a['script']['indic']>0)!=(b['script']['indic']>0)
    v=[float(a['accent_folded']==b['accent_folded'] and bool(at|bt)),
       float(a['core']==b['core'] and bool(at|bt)),
       fuzz.ratio(a['accent_folded'],b['accent_folded'])/100,
       fuzz.token_set_ratio(a['core'],b['core'])/100,
       fuzz.ratio(a['corrected'],b['corrected'])/100,jac(at,bt),overlap,
       float(bool(a['legal']) and a['legal']==b['legal']),
       0.0 if missing else fuzz.token_sort_ratio(x['corrected'],y['corrected'])/100,
       0.0 if missing else jac(ax,bx),jac(an,bn),float(bool(an and bn) and not an&bn),
       jac(al,bl),float(missing),min(len(a['core']),len(b['core']))/max(len(a['core']),len(b['core']),1),
       float(bool(q['country']) and q['country']==t['country']),float(cross),
       b['script']['indic'],float(t['source']==3),float(bin(candidate['mask']).count('1'))]
    v += [float(bool(candidate['mask']&(1<<CHANNELS[ch]))) for ch in CHANNELS]
    v += [candidate['rrf']]
    out=np.asarray(v,dtype=np.float32)
    assert out.shape==(len(FEATURE_NAMES),)
    return out

# SECTION 12
def assign_splits(db,work):
    # DSU over S1 only. rid -> dense S1 slot mapping stays in SQLite.
    work=Path(work);work.mkdir(parents=True,exist_ok=True)
    db.execute('CREATE TABLE IF NOT EXISTS qslot(q INTEGER PRIMARY KEY,slot INTEGER UNIQUE,component INTEGER)')
    db.execute('DELETE FROM qslot')
    rows=db.execute('SELECT q FROM gt_entities ORDER BY q')
    with db:
        db.executemany('INSERT INTO qslot(q,slot) VALUES(?,?)',((q,i) for i,(q,) in enumerate(rows)))
    n=db.execute('SELECT count(*) FROM qslot').fetchone()[0]
    if not n:raise ValueError('Empty training truth')
    parent=np.memmap(work/'parent.i8',mode='w+',dtype=np.int64,shape=(n,))
    for start in range(0,n,100000):parent[start:start+100000]=np.arange(start,min(start+100000,n))
    def root(x):
        while parent[x]!=x:
            parent[x]=parent[parent[x]];x=int(parent[x])
        return x
    last=None;anchor=None
    for t,slot in db.execute('SELECT t.t,s.slot FROM truth t JOIN qslot s ON s.q=t.q ORDER BY t.t,s.slot'):
        if t!=last:anchor=slot;last=t
        else:
            a,b=root(anchor),root(slot)
            parent[max(a,b)]=min(a,b)
    def label(slot):
        v=int.from_bytes(hashlib.blake2b(str(root(slot)).encode(),digest_size=8).digest(),'little')%100
        return 'fit' if v<60 else 'stop' if v<65 else 'cal_fit' if v<75 else 'cal_select' if v<80 else 'tune' if v<90 else 'final'
    cur=db.execute('SELECT q,slot FROM qslot ORDER BY slot')
    while True:
        batch=cur.fetchmany(5000)
        if not batch:break
        with db:
            db.executemany('UPDATE gt_entities SET split=? WHERE q=?',[(label(s),q) for q,s in batch])
            db.executemany('UPDATE qslot SET component=? WHERE q=?',[(root(s),q) for q,s in batch])
    parent.flush()

def select_queries(db,split,limit,seed=42):
    # Keep smallest deterministic random keys: O(limit), independent of full S1 count.
    heap=[]
    for (q,) in db.execute('SELECT q FROM gt_entities WHERE split=?',(split,)):
        rank=int.from_bytes(hashlib.blake2b(f'{seed}:{q}'.encode(),digest_size=8).digest(),'little')
        item=(-rank,-int(q))
        if len(heap)<limit:heapq.heappush(heap,item)
        elif item>heap[0]:heapq.heapreplace(heap,item)
    return sorted(-q for _,q in heap)

def sample_train_candidates(candidates,truth,hard=5,easy=5):
    positives=[c for c in candidates if c['t'] in truth]
    neg=sorted((c for c in candidates if c['t'] not in truth),key=lambda c:(-c['rrf'],c['t']))
    hard_rows=neg[:hard];rest=neg[hard:]
    rest=sorted(rest,key=lambda c:hashlib.blake2b(f"42:{c['q']}:{c['t']}".encode(),digest_size=8).digest())
    return positives+hard_rows+rest[:easy]

def train_model(x,y,xstop,ystop):
    from lightgbm import LGBMClassifier,early_stopping
    model=LGBMClassifier(objective='binary',n_estimators=600,learning_rate=.05,
                         num_leaves=31,min_child_samples=80,max_bin=127,
                         colsample_bytree=.9,reg_lambda=2.0,n_jobs=4,histogram_pool_size=256,
                         random_state=42,deterministic=True,force_col_wise=True,
                         verbosity=-1)
    model.fit(x,y,eval_set=[(xstop,ystop)],eval_metric='binary_logloss',
              callbacks=[early_stopping(40,verbose=False)])
    return model

def pair_groups(db,qids):
    selected=set(map(int,qids))
    mapping={int(q):int(g) for q,g in db.execute('SELECT q,component FROM qslot') if int(q) in selected}
    return np.asarray([mapping[int(q)] for q in qids],dtype=np.int64)

def grouped_oof(x,y,groups,n_splits=3):
    # Optional diagnostic; uses only FIT data and fixed iteration count.
    from lightgbm import LGBMClassifier
    from sklearn.model_selection import GroupKFold
    out=np.empty(len(y),dtype=np.float32)
    for tr,va in GroupKFold(n_splits=n_splits).split(x,y,groups):
        m=LGBMClassifier(n_estimators=250,num_leaves=31,n_jobs=4,random_state=42,verbosity=-1)
        m.fit(x[tr],y[tr]);out[va]=m.predict_proba(x[va])[:,1]
    return out

def mine_fit_only(db,model,qids,get_candidates,feature_fn,cap=100000):
    mined=[]
    for q in qids:
        if db.execute('SELECT split FROM gt_entities WHERE q=?',(q,)).fetchone()[0]!='fit':
            raise ValueError('Hard negative mining may only read FIT labels')
        cs=get_candidates(q);truth=truth_for(db,q)
        if not cs:continue
        x=np.vstack([feature_fn(c) for c in cs]);p=model.predict_proba(x)[:,1]
        for j in sorted(range(len(cs)),key=lambda i:(-p[i],cs[i]['t'])):
            if cs[j]['t'] not in truth:
                mined.append((x[j],0))
                break
        if len(mined)>=cap:break
    return mined

# SECTION 13
class Calibrator:
    def __init__(self,kind):self.kind=kind;self.model=None
    def fit(self,p,y):
        if len(np.unique(y))<2:raise ValueError('Calibration needs both classes')
        if self.kind=='isotonic':
            from sklearn.isotonic import IsotonicRegression
            self.model=IsotonicRegression(out_of_bounds='clip').fit(p,y)
        elif self.kind=='platt':
            from sklearn.linear_model import LogisticRegression
            z=np.log(np.clip(p,1e-6,1-1e-6)/np.clip(1-p,1e-6,1-1e-6)).reshape(-1,1)
            self.model=LogisticRegression(C=1000,max_iter=300).fit(z,y)
        else:raise ValueError(self.kind)
        return self
    def predict(self,p):
        p=np.asarray(p)
        if self.kind=='isotonic':return self.model.predict(p)
        z=np.log(np.clip(p,1e-6,1-1e-6)/np.clip(1-p,1e-6,1-1e-6)).reshape(-1,1)
        return self.model.predict_proba(z)[:,1]

def choose_calibrator(p_fit,y_fit,p_select,y_select):
    from sklearn.metrics import brier_score_loss
    models=[Calibrator(k).fit(p_fit,y_fit) for k in ['platt','isotonic']]
    losses=[brier_score_loss(y_select,m.predict(p_select)) for m in models]
    # Selection labels never fit either calibrator, and never tune the match threshold.
    j=min(range(2),key=lambda i:(losses[i],i))
    return models[j],dict(zip(['platt','isotonic'],losses))

# SECTION 14
def threshold_search(entity_factory,thresholds=None):
    # Factory yields (q, FULL true set, [(target,probability),...]) for EVERY tune q.
    grid=np.asarray(thresholds if thresholds is not None else np.linspace(0,1,1001),dtype=float)
    grid=np.unique(np.r_[grid,np.nextafter(1.0,2.0)]) # include predict-empty endpoint
    delta=np.zeros(len(grid)+1,dtype=np.float64);base=0.0;n=0
    for q,true,scored in entity_factory():
        if len({t for t,p in scored})!=len(scored):raise ValueError('Duplicate candidate')
        if any(not math.isfinite(p) or p<0 or p>1 for t,p in scored):raise ValueError('Invalid probability')
        m=len(true);empty=1.0 if not m else 0.0
        base+=empty;n+=1
        ordered=sorted(scored,key=lambda x:(-x[1],x[0]));tp=0;accepted=0;previous=empty
        pos=0
        while pos<len(ordered):
            score=ordered[pos][1]
            while pos<len(ordered) and ordered[pos][1]==score:
                tp+=ordered[pos][0] in true;accepted+=1;pos+=1
            value=(1.25*tp/(accepted+.25*m)) if accepted+.25*m else 1.0
            # This transition contributes to every threshold <= score.
            end=int(np.searchsorted(grid,score,side='right'))
            delta[0]+=value-previous;delta[end]-=value-previous;previous=value
    if not n:raise ValueError('Empty threshold-tuning universe')
    scores=(base+np.cumsum(delta[:-1]))/n
    top=np.flatnonzero(np.isclose(scores,scores.max(),rtol=0,atol=1e-12))
    j=int(top[-1]) # conservative deterministic tie break
    return float(grid[j]),float(scores[j]),grid,scores

# SECTION 15
def decide(scored,threshold):
    if len({t for t,p in scored})!=len(scored):raise ValueError('Duplicate scored targets')
    return {int(t) for t,p in scored if p>=threshold}

# SECTION 16
def entity_f05(true,pred):
    true=set(true);pred=set(pred)
    if not true and not pred:return 1.0
    tp=len(true&pred)
    return 1.25*tp/(len(pred)+.25*len(true)) if pred or true else 1.0

def oracle_f05(true,candidates):
    m=len(true);h=len(set(true)&set(candidates))
    return 1.0 if m==0 else 5*h/(m+4*h)

def evaluate_entities(entity_factory,threshold,slice_fn=None):
    sums=defaultdict(lambda:Counter(n=0,score=0.0,oracle=0.0,hit=0,true=0,complete=0))
    for q,true,scored in entity_factory():
        cand={int(t) for t,p in scored}
        if len(cand)!=len(scored):raise ValueError('Duplicate target in candidate set')
        pred=decide(scored,threshold)
        tags=['all']+(list(slice_fn(q)) if slice_fn else [])
        for tag in tags:
            s=sums[tag];s['n']+=1;s['score']+=entity_f05(true,pred)
            s['oracle']+=oracle_f05(true,cand);s['hit']+=len(true&cand);s['true']+=len(true)
            if true:s['nonempty']+=1;s['complete']+=true<=cand
    return {tag:dict(n=s['n'],macro_f05=s['score']/s['n'],oracle=s['oracle']/s['n'],
                     pair_recall=s['hit']/s['true'] if s['true'] else None,
                     complete_recall=s['complete']/s['nonempty'] if s['nonempty'] else None)
            for tag,s in sums.items()}

def difficulty_tags(db,q):
    a=fetch_record(db,q);vendors=[fetch_record(db,t) for t in truth_for(db,q)]
    tags=['country:'+a['country'],'singleton' if not vendors else 'non_singleton']
    if any(not r['address']['has'] for r in vendors):tags.append('true_vendor_missing_address')
    if any(r['name']['script']['indic']>0 for r in vendors):tags.append('true_vendor_indic')
    if len(vendors)==1:tags.append('one_match')
    if len(vendors)>1:tags.append('many_matches')
    return tags

def metric_tests():
    assert math.isclose(entity_f05({'a','b','c'},{'a','b'}),10/11)
    assert math.isclose(oracle_f05({'a','b','c'},{'a','b'}),10/11)
    assert entity_f05(set(),set())==1
    assert entity_f05(set(),{'x'})==0
    assert entity_f05({'x'},set())==0
    rows=[(1,{1,2,3},[(1,.9),(2,.8)]),(2,set(),[]),(3,{7},[])]
    report=evaluate_entities(lambda:iter(rows),.5)['all']
    assert math.isclose(report['macro_f05'],(10/11+1)/3)
    tau,score,grid,scores=threshold_search(lambda:iter(rows),[0,.5,.8,.9,1])
    brute=[sum(entity_f05(t,decide(s,g)) for q,t,s in rows)/len(rows) for g in grid]
    assert np.allclose(scores,brute)
    assert missing_clean('<NULL>')==''
    assert not tokens(clean_address('<NULL>'))
    assert clean_name('#hanumaneducational')=='hanumaneducational'
    assert clean_name('#98587')=='98587'

# SECTION 17
def external_id(db,rid):
    row=db.execute('SELECT eid FROM records WHERE rid=?',(int(rid),)).fetchone()
    if row is None:raise KeyError(rid)
    return row[0]

def write_outputs(db,query_ids,score_query,threshold,outdir):
    outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True)
    paths=[outdir/'matching_results.tsv',outdir/'candidate_pairs.tsv']
    temps=[Path(str(p)+'.tmp') for p in paths]
    with open(temps[0],'w',encoding='utf-8',newline='') as fm,open(temps[1],'w',encoding='utf-8',newline='') as fc:
        wm,wc=csv.writer(fm,delimiter='\t'),csv.writer(fc,delimiter='\t')
        wm.writerow(['source1_entity_id','matched_entity_ids'])
        wc.writerow(['source1_entity_id','candidate_entity_ids'])
        for q in query_ids:
            scored=score_query(int(q));candidates={int(t) for t,p in scored}
            matches=decide(scored,threshold)
            if not matches<=candidates:raise AssertionError('Output invariant')
            for t in candidates:
                row=db.execute('SELECT source FROM records WHERE rid=?',(t,)).fetchone()
                if row is None or row[0] not in (2,3):raise ValueError('Invalid target')
            eid=external_id(db,q)
            wm.writerow([eid,','.join(sorted(external_id(db,t) for t in matches))])
            wc.writerow([eid,','.join(sorted(external_id(db,t) for t in candidates))])
    for tmp,path in zip(temps,paths):os.replace(tmp,path)
    return paths

# SECTION 18
def validate_outputs(db,match_path,candidate_path):
    db.execute('CREATE TEMP TABLE IF NOT EXISTS seen_output(q INTEGER PRIMARY KEY)')
    db.execute('DELETE FROM seen_output')
    from itertools import zip_longest
    with open(match_path,encoding='utf8',newline='') as fm,open(candidate_path,encoding='utf8',newline='') as fc:
        rm=csv.DictReader(fm,delimiter='\t');rc=csv.DictReader(fc,delimiter='\t')
        if rm.fieldnames!=['source1_entity_id','matched_entity_ids']:raise ValueError('Match header')
        if rc.fieldnames!=['source1_entity_id','candidate_entity_ids']:raise ValueError('Candidate header')
        for m,c in zip_longest(rm,rc):
            if m is None or c is None:raise ValueError('Row count mismatch')
            if m['source1_entity_id']!=c['source1_entity_id']:raise ValueError('Row alignment')
            q=lookup_id(db,m['source1_entity_id'],{1})
            db.execute('INSERT INTO seen_output VALUES(?)',(q,))
            ms=m['matched_entity_ids'].split(',') if m['matched_entity_ids'] else []
            cs=c['candidate_entity_ids'].split(',') if c['candidate_entity_ids'] else []
            if len(ms)!=len(set(ms)) or len(cs)!=len(set(cs)):raise ValueError('Duplicate targets')
            if not set(ms)<=set(cs):raise ValueError('Matches outside candidates')
            for eid in cs:lookup_id(db,eid,{2,3})
    missing=db.execute('SELECT count(*) FROM records r WHERE source=1 AND NOT EXISTS '
                       '(SELECT 1 FROM seen_output s WHERE s.q=r.rid)').fetchone()[0]
    if missing:raise ValueError(f'{missing} S1 output rows missing')
    db.commit()

def official_validate(script,test_dir,output_dir):
    import subprocess,sys
    subprocess.run([sys.executable,str(script),'--matching',str(Path(output_dir)/'matching_results.tsv'),
                    '--candidate',str(Path(output_dir)/'candidate_pairs.tsv'),
                    '--test-dir',str(test_dir),'--check-ids'],check=True)

# SECTION 19
def build_all_indexes(db,root,kinds=('char','rare'),shard=100000):
    metas=[]
    for country,source in db.execute('SELECT DISTINCT country,source FROM records WHERE source IN (2,3) ORDER BY country,source').fetchall():
        key=hashlib.sha256(country.encode()).hexdigest()[:16]
        for kind in kinds:
            path=Path(root)/f'{key}-s{source}-{kind}'
            if kind.startswith('minhash_'):
                metas.append(build_minhash_index(db,path,country,source,kind.split('_',1)[1],shard=min(shard,50000)))
            else:metas.append(make_index(db,path,country,source,kind,shard=shard))
    return metas

def retrieve_batch(db,metas,queries,k=60,country_verified=False):
    # Queries in this function share country to avoid redundant filtering.
    if not queries:return {}
    if len({q['country'] for q in queries})!=1:raise ValueError('Batch by query country')
    countries=route_countries(queries[0]['country'],{m['country'] for m in metas},country_verified)
    by_q={q['rid']:[] for q in queries}
    for q in queries:
        hits,overflow=exact_hits(db,q,countries)
        by_q[q['rid']].extend(hits)
        q['exact_overflow']=overflow
    for meta in metas:
        if meta['country'] not in countries:continue
        hits=(query_minhash_index(db,meta,queries,k=BUDGETS[meta['kind']]) if meta['kind'].startswith('minhash_')
              else query_index(meta,queries,k=BUDGETS[meta['kind']]))
        for hit in hits:by_q[hit.q].append(hit)
    for query in queries:query['precap_targets']={h.t for h in by_q[query['rid']]}
    return {q:fuse(hits,k=k) for q,hits in by_q.items()}

def iter_candidate_batches(db,metas,qids=None,k=60,country_verified=False,batch_size=128):
    # Disk selection table bounds query inventory even for full test inference.
    db.execute('CREATE TEMP TABLE IF NOT EXISTS selected_q(q INTEGER PRIMARY KEY)')
    db.execute('DELETE FROM selected_q')
    if qids is None:db.execute('INSERT INTO selected_q SELECT rid FROM records WHERE source=1')
    else:db.executemany('INSERT INTO selected_q VALUES(?)',((int(q),) for q in qids))
    countries=[r[0] for r in db.execute('SELECT DISTINCT country FROM records r JOIN selected_q s ON s.q=r.rid ORDER BY country')]
    for country in countries:
        cursor=db.execute('SELECT r.rid FROM records r JOIN selected_q s ON s.q=r.rid WHERE country=? ORDER BY r.rid',(country,))
        while True:
            ids=cursor.fetchmany(batch_size)
            if not ids:break
            records=[fetch_record(db,r[0]) for r in ids]
            cs=retrieve_batch(db,metas,records,k,country_verified)
            yield records,cs

def build_pair_file(db,metas,qids,path,training=False,cap=1500000,k=60,country_verified=False):
    import pyarrow as pa,pyarrow.parquet as pq
    schema=pa.schema([('q',pa.int64()),('t',pa.int64()),('y',pa.int8())]+
                     [(name,pa.float32()) for name in FEATURE_NAMES])
    total=0;buffer=[]
    def flush(writer):
        nonlocal buffer
        if buffer:writer.write_table(pa.Table.from_pylist(buffer,schema=schema));buffer=[]
    with pq.ParquetWriter(path,schema,compression='zstd') as writer:
        for records,by_q in iter_candidate_batches(db,metas,qids,k,country_verified):
            target_cache=fetch_many(db,[c['t'] for cs in by_q.values() for c in cs])
            for q in records:
                truth=truth_for(db,q['rid']);cs=by_q[q['rid']]
                if training:cs=sample_train_candidates(cs,truth)
                if total+len(cs)>cap:raise MemoryError('Pair cap exceeded: reduce FIT query count, never truncate validation candidates')
                for c in cs:
                    t=target_cache[c['t']];v=pair_features(q,t,c)
                    buffer.append(dict(q=q['rid'],t=c['t'],y=int(c['t'] in truth),**dict(zip(FEATURE_NAMES,map(float,v)))))
                    total+=1
                    if len(buffer)>=LIMITS['feature_batch']:flush(writer)
        flush(writer)
    return total

def read_pair_file(path,cap=1500000):
    import pyarrow.parquet as pq
    pf=pq.ParquetFile(path);n=pf.metadata.num_rows
    if n>cap:raise MemoryError('Training allocation cap exceeded')
    x=np.empty((n,len(FEATURE_NAMES)),np.float32);y=np.empty(n,np.int8)
    q=np.empty(n,np.int64);t=np.empty(n,np.int64);offset=0
    for batch in pf.iter_batches(batch_size=8192):
        d=batch.to_pydict();b=len(d['q']);sl=slice(offset,offset+b)
        x[sl]=np.column_stack([d[f] for f in FEATURE_NAMES]);y[sl]=d['y'];q[sl]=d['q'];t[sl]=d['t'];offset+=b
    return x,y,q,t

def scored_factory(db,qids,q,t,p):
    # Arrays are bounded validation samples, NOT whole train/test populations.
    order=np.lexsort((t,q));qs=q[order];ts=t[order];ps=np.asarray(p)[order]
    def factory():
        for query in qids:
            lo,hi=np.searchsorted(qs,query,'left'),np.searchsorted(qs,query,'right')
            yield int(query),truth_for(db,query),[(int(a),float(b)) for a,b in zip(ts[lo:hi],ps[lo:hi])]
    return factory

def fit_and_validate(db,paths,qsets):
    # All paths are complete sampled-query candidate files except FIT negatives.
    x,y,_,_=read_pair_file(paths['fit']);xs,ys,_,_=read_pair_file(paths['stop'])
    model=train_model(x,y,xs,ys);del x,y,xs,ys
    xc,yc,_,_=read_pair_file(paths['cal_fit']);pc=model.predict_proba(xc)[:,1];del xc
    xv,yv,_,_=read_pair_file(paths['cal_select']);pv=model.predict_proba(xv)[:,1];del xv
    cal,losses=choose_calibrator(pc,yc,pv,yv);del pc,yc,pv,yv
    xt,yt,q,t=read_pair_file(paths['tune']);p=cal.predict(model.predict_proba(xt)[:,1]);del xt,yt
    factory=scored_factory(db,qsets['tune'],q,t,p);tau,score,_,_=threshold_search(factory)
    del q,t,p,factory
    xf,yf,q,t=read_pair_file(paths['final']);p=cal.predict(model.predict_proba(xf)[:,1]);del xf,yf
    final=evaluate_entities(scored_factory(db,qsets['final'],q,t,p),tau,lambda q: difficulty_tags(db,q))
    return model,cal,tau,dict(calibration_brier=losses,tune_f05=score,final=final)

def infer(db,metas,model,cal,threshold,outdir,k=60,country_verified=False):
    # Incremental files; score each final fused candidate exactly once.
    outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True)
    pm=outdir/'matching_results.tsv';pc=outdir/'candidate_pairs.tsv'
    with open(str(pm)+'.tmp','w',encoding='utf8',newline='') as fm,open(str(pc)+'.tmp','w',encoding='utf8',newline='') as fc:
        wm=csv.writer(fm,delimiter='\t');wc=csv.writer(fc,delimiter='\t')
        wm.writerow(['source1_entity_id','matched_entity_ids']);wc.writerow(['source1_entity_id','candidate_entity_ids'])
        for records,by_q in iter_candidate_batches(db,metas,None,k,country_verified):
            target_cache=fetch_many(db,[c['t'] for cs in by_q.values() for c in cs])
            vectors=[pair_features(q,target_cache[c['t']],c) for q in records for c in by_q[q['rid']]]
            probs=cal.predict(model.predict_proba(np.vstack(vectors))[:,1]) if vectors else np.empty(0)
            offset=0
            for q in records:
                cs=by_q[q['rid']]
                scored=[(c['t'],float(p)) for c,p in zip(cs,probs[offset:offset+len(cs)])]
                offset+=len(cs)
                targets={c['t'] for c in cs};matches=decide(scored,threshold)
                assert matches<=targets
                eid=external_id(db,q['rid'])
                wc.writerow([eid,','.join(sorted(external_id(db,t) for t in targets))])
                wm.writerow([eid,','.join(sorted(external_id(db,t) for t in matches))])
    os.replace(str(pm)+'.tmp',pm);os.replace(str(pc)+'.tmp',pc)
    validate_outputs(db,pm,pc)

# SECTION 20
def benchmark(db,metas,qids,k=60,country_verified=False):
    import resource
    start=time.perf_counter();count=0;queries=0;oracle=0.;preoracle=0.;hit=0;true_total=0;hist=Counter();exact_overflow=0
    for records,by_q in iter_candidate_batches(db,metas,qids,k,country_verified):
        for q in records:
            cand={c['t'] for c in by_q[q['rid']]};truth=truth_for(db,q['rid'])
            queries+=1;count+=len(cand);hist[len(cand)]+=1
            oracle+=oracle_f05(truth,cand);preoracle+=oracle_f05(truth,q['precap_targets'])
            exact_overflow+=int(q.get('exact_overflow',False));hit+=len(truth&cand);true_total+=len(truth)
    elapsed=time.perf_counter()-start
    def percentile(frac):
        total=0
        for size,n in sorted(hist.items()):
            total+=n
            if total>=max(1,math.ceil(queries*frac)):return size
        return 0
    return dict(queries=queries,pairs=count,seconds=elapsed,
                queries_per_second=queries/max(elapsed,1e-9),
                oracle=oracle/queries if queries else None,precap_oracle=preoracle/queries if queries else None,
                mean_k=count/queries if queries else 0,p95_k=percentile(.95),p99_k=percentile(.99),
                max_k=max(hist,default=0),exact_overflows=exact_overflow,
                pair_recall=hit/true_total if true_total else None,
                k_hist=dict(hist),maxrss_platform_units=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

def budget_curve(db,metas,qids,values=(5,10,20,40,60,100),country_verified=False):
    return [dict(k=k,**benchmark(db,metas,qids,k,country_verified)) for k in values]

def core_run(train_dir,test_dir,work,out,kinds=('char','rare','translit'),k=60,workers=1,country_verified=False):
    import pickle
    work=Path(work);work.mkdir(parents=True,exist_ok=True)
    train=connect(work/'train.sqlite')
    if train.execute('SELECT count(*) FROM records').fetchone()[0]:raise ValueError('Use fresh work directory')
    for source in (1,2,3):ingest(train,Path(train_dir)/f'train_source{source}.tsv',source,workers=workers)
    load_truth(train,Path(train_dir)/'train_ground_truth.tsv');assign_splits(train,work/'groups')
    rules,rule_report=learn_spelling(train);apply_spelling(train,rules)
    configure_feature_weights(train,work/'feature_idf.npy')
    # Cross-country evidence is diagnosed; hard partitioning remains opt-in until verified.
    metas=build_all_indexes(train,work/'train_index',kinds)
    limits={'fit':50000,'stop':5000,'cal_fit':10000,'cal_select':5000,'tune':10000,'final':10000}
    qsets={split:select_queries(train,split,n) for split,n in limits.items()}
    if any(not q for q in qsets.values()):raise ValueError('Every split needs data; use larger development sample')
    paths={s:str(work/f'{s}.parquet') for s in qsets}
    for split,ids in qsets.items():build_pair_file(train,metas,ids,paths[split],split=='fit',k=k,country_verified=country_verified)
    model,cal,tau,report=fit_and_validate(train,paths,qsets)
    report.update(spelling=rule_report,dependencies=dependency_manifest(),k=k,kinds=list(kinds),country_verified=country_verified)
    (work/'report.json').write_text(json.dumps(report,indent=2))
    with open(work/'model.pkl','wb') as f:pickle.dump((model,cal,tau,rules,kinds,k),f)
    train.close()
    test=connect(work/'test.sqlite')
    for source in (1,2,3):ingest(test,Path(test_dir)/f'test_source{source}.tsv',source,rules,workers)
    metas=build_all_indexes(test,work/'test_index',kinds)
    infer(test,metas,model,cal,tau,out,k,country_verified)
    test.close()
    return report

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--train');parser.add_argument('--test')
    parser.add_argument('--work',default='work');parser.add_argument('--output',default='output')
    args=parser.parse_args()
    if args.self_test:metric_tests();print('Core regression tests passed')
    elif args.train and args.test:print(json.dumps(core_run(args.train,args.test,args.work,args.output,kinds=('char','rare','translit'),k=60,workers=8),indent=2))
    else:parser.error('Supply --self-test or --train and --test')
