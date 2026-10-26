import csv, gzip, json, math, random, re
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "data" / "msmarco"
OUT = PROJECT_ROOT / "results" / "query_channel"
PLOT = OUT / "图表"
SEED = 20260914
MAX_Q = 12000
TOKEN_RE = re.compile(r"[a-z0-9]+")

def toks(s): return TOKEN_RE.findall((s or "").lower())

def reservoir_records(path, limit):
    rng = random.Random(SEED); sample=[]; seen=0; malformed=0; empty=0; qids=set(); dup=0
    with open(path, encoding="utf-8") as f:
        for line in f:
            try: r=json.loads(line)
            except Exception: malformed += 1; continue
            seen += 1; qid=str(r.get("query_id", ""))
            if qid in qids: dup += 1
            qids.add(qid)
            pos=r.get("positive_passages") or []; neg=r.get("negative_passages") or []
            if not r.get("query") or not pos or not neg: empty += 1; continue
            item={"query_id":qid,"query":r["query"],"positive":pos[0],"negative":neg[:3]}
            if len(sample)<limit: sample.append(item)
            else:
                j=rng.randrange(seen)
                if j < limit: sample[j]=item
    audit={"file":str(path),"records_scanned":seen,"eligible_records":len(sample),"malformed":malformed,"missing_or_empty_candidates":empty,"duplicate_query_ids":dup}
    return sample,audit

def split(items):
    rng=random.Random(SEED); rng.shuffle(items)
    n=len(items); a=int(n*.7); b=int(n*.15)
    return items[:a],items[a:a+b],items[a+b:]

def build_idf(items):
    df=Counter(); N=len(items)
    for x in items:
        for t in set(toks(x["query"])): df[t]+=1
    return {t:math.log((N+1)/(v+1))+1 for t,v in df.items()}

def semantic_message(query, idf, k):
    # Query keywords are ranked by informativeness, then alphabetically for determinism.
    ts=set(toks(query)); ranked=sorted(ts,key=lambda t:(-idf.get(t,1.0),t))
    return tuple(ranked[:k])

def channel(message, keep, vocab, rng):
    out=[]
    for t in message:
        u=rng.random()
        if u < keep: out.append(t)
        elif u < keep + (1-keep)*.35: out.append("<eps>")
        else: out.append(rng.choice(vocab))
    return tuple(out)

def score(obs, passage):
    a=set(t for t in obs if t != "<eps>"); b=set(toks((passage.get("title","")+" "+passage.get("text", ""))))
    return len(a & b) / max(1,len(a))

def entropy(c, n):
    return -sum(v/n*math.log2(v/n) for v in c.values() if v)

def mi(xs, ys):
    n=len(xs); cx=Counter(xs); cy=Counter(ys); cxy=Counter(zip(xs,ys))
    return sum(v/n*math.log2(v*n/(cx[x]*cy[y])) for (x,y),v in cxy.items())

def evaluate(items, idf, k, bob_keep, eve_keep, seed):
    rng=random.Random(seed); vocab=sorted({t for x in items for t in toks(x["query"])})[:2000] or ["noise"]
    S=[]; Y=[]; Z=[]; bob_correct=eve_correct=0; bob_rr=eve_rr=0.0
    for x in items:
        s=semantic_message(x["query"],idf,k); y=channel(s,bob_keep,vocab,rng); z=channel(s,eve_keep,vocab,rng)
        candidates=[x["positive"]]+x["negative"]
        bs=sorted(range(len(candidates)),key=lambda i:(-score(y,candidates[i]),i)); es=sorted(range(len(candidates)),key=lambda i:(-score(z,candidates[i]),i))
        br=bs.index(0)+1; er=es.index(0)+1
        bob_correct += br==1; eve_correct += er==1; bob_rr += 1/br; eve_rr += 1/er
        S.append(s); Y.append(y); Z.append(z)
    return {"queries":len(items),"k_tokens":k,"cost_tokens":sum(map(len,S))/len(S),"bob_accuracy":bob_correct/len(S),"eve_accuracy":eve_correct/len(S),"bob_mrr_local":bob_rr/len(S),"eve_mrr_local":eve_rr/len(S),"D_bob":1-bob_correct/len(S),"D_eve":1-eve_correct/len(S),"I_S_Bob_bits":mi(S,Y),"I_S_Eve_bits":mi(S,Z),"secrecy_advantage_bits":mi(S,Y)-mi(S,Z)}

def write_csv(path, rows):
    with open(path,"w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    PLOT.mkdir(parents=True, exist_ok=True)
    items,audit=reservoir_records(ROOT/"train.jsonl",MAX_Q)
    train,val,test=split(items); idf=build_idf(train)
    with open(OUT/"MSMARCO_数据审计.txt","w",encoding="utf-8") as f:
        f.write("MS MARCO query-local experiment audit\n")
        for k,v in audit.items(): f.write(f"{k}: {v}\n")
        f.write(f"split_train: {len(train)}\nsplit_validation: {len(val)}\nsplit_test: {len(test)}\n")
        f.write("Scope: query-local positive-vs-negative passage ranking; no full-corpus metrics.\n")
    write_csv(OUT/"MSMARCO_数据划分.csv",[{"split":"train","queries":len(train)},{"split":"validation","queries":len(val)},{"split":"test","queries":len(test)}])
    rows=[]
    conditions={"Bob-better":(.90,.55),"Equal":(.75,.75),"Eve-better":(.55,.90)}
    for cond,(bk,ek) in conditions.items():
        for k in (2,4,8):
            r=evaluate(test,idf,k,bk,ek,SEED+k); r.update({"condition":cond,"bob_keep":bk,"eve_keep":ek}); rows.append(r)
    write_csv(OUT/"MSMARCO_BobEve信道实验结果.csv",rows)
    with open(OUT/"MSMARCO_实验结果摘要.txt","w",encoding="utf-8") as f:
        f.write("MS MARCO query-local semantic communication experiment\n\n")
        f.write("Task: rank one supplied positive passage against up to three supplied negatives.\n")
        f.write("D_bob=1-local top-1 accuracy; cost=mean transmitted keyword count; secrecy=I(S;Y)-I(S;Z).\n")
        for r in rows: f.write(json.dumps(r,ensure_ascii=False)+"\n")
    def svg_line(path, ykey, ylabel):
        width,height,left,right,top,bottom=780,470,85,30,35,70
        values=[r[ykey] for r in rows]; lo,hi=min(values),max(values)
        if hi==lo: hi=lo+1
        colors={"Bob-better":"#1f4e79","Equal":"#666666","Eve-better":"#b85450"}
        def sx(x): return left+(x-2)/(8-2)*(width-left-right)
        def sy(y): return height-bottom-(y-lo)/(hi-lo)*(height-top-bottom)
        a=['<svg xmlns="http://www.w3.org/2000/svg" width="780" height="470" viewBox="0 0 780 470">','<rect width="100%" height="100%" fill="white"/>']
        for j in range(5):
            yy=top+j*(height-top-bottom)/4; val=hi-j*(hi-lo)/4
            a.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{width-right}" y2="{yy:.1f}" stroke="#d9d9d9"/>')
            a.append(f'<text x="{left-8}" y="{yy+4:.1f}" text-anchor="end" font-family="Arial" font-size="12">{val:.3f}</text>')
        a.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#222"/><line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#222"/>')
        for x in (2,4,8): a.append(f'<text x="{sx(x):.1f}" y="{height-bottom+22}" text-anchor="middle" font-family="Arial" font-size="12">{x}</text>')
        a.append(f'<text x="{width/2}" y="{height-20}" text-anchor="middle" font-family="Arial" font-size="14">Semantic message length k (tokens)</text>')
        a.append(f'<text transform="translate(18 {height/2}) rotate(-90)" text-anchor="middle" font-family="Arial" font-size="14">{ylabel}</text>')
        for i,cond in enumerate(conditions):
            q=sorted([r for r in rows if r["condition"]==cond],key=lambda r:r["k_tokens"]); pts=' '.join(f'{sx(r["k_tokens"]):.1f},{sy(r[ykey]):.1f}' for r in q); c=colors[cond]
            a.append(f'<polyline points="{pts}" fill="none" stroke="{c}" stroke-width="2.5"/>')
            for r in q: a.append(f'<circle cx="{sx(r["k_tokens"]):.1f}" cy="{sy(r[ykey]):.1f}" r="4" fill="{c}"/>')
            a.append(f'<rect x="{width-165}" y="{top+i*22-10}" width="9" height="9" fill="{c}"/><text x="{width-151}" y="{top+i*22}" font-family="Arial" font-size="12">{cond}</text>')
        a.append('</svg>'); path.write_text('\n'.join(a),encoding="utf-8")
    svg_line(PLOT/"MSMARCO_Bob局部准确率.svg","bob_accuracy","Bob local top-1 accuracy")
    svg_line(PLOT/"MSMARCO_保密性优势.svg","secrecy_advantage_bits","Secrecy advantage I(S;Y)-I(S;Z) (bits)")
    print("OUTPUT",OUT); print("AUDIT",audit); print("TEST",len(test))
    for r in rows: print(r)

if __name__=="__main__": main()
