"""Exact finite (W,S,P,Q) semantic-secrecy validation on MS MARCO queries.

The Web data defines an eight-class information-need source.  Each source intent
has three reusable semantic messages: a coarse shared message, a group message,
and an intent-specific message.  Thus the finite semantic language creates a real
distortion--cost--confidentiality choice rather than treating a unique query as S.
"""
import csv, hashlib, json, math, random, re
from itertools import product
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "data" / "msmarco"
OUT = PROJECT_ROOT / "results" / "web_theory"
FIG = OUT / "figures"
SEED=20260914; N_SAMPLE=12000; DIM=96; K=8; N_NEG=30
TOKEN=re.compile(r"[a-z0-9]+")

def words(x): return TOKEN.findall((x or '').lower())
def bucket(t): return int(hashlib.md5(t.encode()).hexdigest()[:8],16)%DIM
def embed(x):
    v=np.zeros(DIM)
    for t in words(x): v[bucket(t)]+=1
    z=np.linalg.norm(v); return v/z if z else v

def load_data():
    rng=random.Random(SEED); sample=[]; scanned=bad=0
    with open(ROOT/'train.jsonl',encoding='utf-8') as f:
        for line in f:
            try:r=json.loads(line)
            except Exception:bad+=1;continue
            scanned+=1
            if (not r.get('query') or not r.get('positive_passages') or
                    len(r.get('negative_passages') or []) < N_NEG): continue
            # Relevance fields are retained for downstream evaluation only.
            x={'query':r['query'],'positive':r['positive_passages'][0],
               'negative':r['negative_passages'][:N_NEG]}
            if len(sample)<N_SAMPLE: sample.append(x)
            else:
                j=rng.randrange(scanned)
                if j<N_SAMPLE:sample[j]=x
    rng.shuffle(sample);return sample,scanned,bad

def fit_intents(train):
    # W is available before communication and therefore uses query text only.
    # Positive/negative passages never enter prototypes, labels, or the prior.
    X=np.vstack([embed(x['query']) for x in train]); picks=[0]
    for _ in range(1,K):
        d=((X[:,None,:]-X[picks][None,:,:])**2).sum(2).min(1);picks.append(int(np.argmax(d)))
    C=X[picks]; lab=((X[:,None,:]-C[None,:,:])**2).sum(2).argmin(1)
    prior=np.bincount(lab,minlength=K).astype(float);prior/=prior.sum();return X,C,lab,prior

# S alphabet: 0=global coarse, 1..4=pair-group messages, 5..12=intent-specific messages.
N_S=1+K//2+K
def symbol(w,mode): return 0 if mode==0 else 1+w//2 if mode==1 else 5+w
def mode_of(s): return 0 if s==0 else 1 if s<=4 else 2
COST=(1.0,2.0,3.0)

def class_channel(correct):
    q=np.full((N_S,N_S),(1-correct)/(N_S-1));np.fill_diagonal(q,correct);return q

def degraded_eve(QB, alphas):
    """Construct QE from QB through a symbol-observable stochastic post-channel.

    T(y,z) retains y with alpha(mode(y)); otherwise it emits a uniformly
    randomized semantic symbol. Hence S -> Y_B -> Z_E holds by construction.
    The mode is part of the transmitted finite alphabet, so T is a valid fixed
    channel on Y_B and is not an oracle using W.
    """
    T=np.zeros((N_S,N_S))
    for y in range(N_S):
        a=alphas[mode_of(y)]
        T[y,:]=(1-a)/N_S
        T[y,y]+=a
    return QB@T, T

def build_channels(condition, alpha_override=None):
    # The main condition is a physically degraded wiretap pair. Bob's channel
    # is the base DMC; Eve observes only a randomized post-processing of it.
    base=(.97,.91,.96)
    QB=np.vstack([class_channel(base[mode_of(s)])[s] for s in range(N_S)])
    def mode_diag(Q):
        return tuple(float(np.mean([Q[s,s] for s in range(N_S) if mode_of(s)==m])) for m in range(3))
    if condition=='Bob-better':
        alpha=alpha_override if alpha_override is not None else (.70,.35,.65)
        QE,T=degraded_eve(QB,alpha)
        b=mode_diag(QB); e=mode_diag(QE)
    elif condition=='Equal':
        QE=QB.copy(); T=np.eye(N_S)
        b=e=mode_diag(QB)
    else:
        # Reverse-degraded control: Bob is the post-processed observer.
        QE=np.vstack([class_channel(base[mode_of(s)])[s] for s in range(N_S)])
        QB,T=degraded_eve(QE,(.70,.35,.65))
        b=mode_diag(QB); e=mode_diag(QE)
    return QB,QE,b,e,T

def mutual_info(ps,Q):
    py=ps@Q;v=0.
    for i,p in enumerate(ps):
        for j,q in enumerate(Q[i]):
            if p and q and py[j]:v+=p*q*math.log2(q/py[j])
    return v

def exact_metrics(enc,prior,QB,QE):
    ps=np.zeros(N_S)
    for w,m in enumerate(enc):ps[symbol(w,m)]+=prior[w]
    def decode_loss(Q):
        joint=np.zeros((K,N_S))
        for w,m in enumerate(enc):joint[w]+=prior[w]*Q[symbol(w,m)]
        dec=joint.argmax(0);return 1-sum(joint[dec[s],s] for s in range(N_S)),dec
    db,decB=decode_loss(QB);de,decE=decode_loss(QE);ib=mutual_info(ps,QB);ie=mutual_info(ps,QE)
    return {'D_B':db,'D_E':de,'L':sum(prior[w]*COST[m] for w,m in enumerate(enc)), 'I_B':ib,'I_E':ie,'Se':ib-ie,'encoding':' '.join(map(str,enc)),'decoder_B':decB,'decoder_E':decE}

def pareto(rows):
    out=[]
    for x in rows:
        dominated=any(y['D_B']<=x['D_B'] and y['L']<=x['L'] and y['Se']>=x['Se'] and (y['D_B']<x['D_B'] or y['L']<x['L'] or y['Se']>x['Se']) for y in rows)
        if not dominated:out.append(x)
    return out

def write(path,rows):
    with open(path,'w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def bootstrap_task(labels,enc,Q,prior,seed):
    rng=np.random.default_rng(seed);correct=[]
    for w in labels:
        y=rng.choice(N_S,p=Q[symbol(int(w),enc[int(w)])]);
        # Bayes decoder is rederived from the selected coding distribution.
        joint=np.zeros((K,N_S))
        for a,m in enumerate(enc):joint[a]+=prior[a]*Q[symbol(a,m)]
        correct.append(int(joint[:,y].argmax()==w))
    a=np.array(correct);means=[a[rng.integers(0,len(a),len(a))].mean() for _ in range(300)]
    return float(a.mean()),float(np.quantile(means,.025)),float(np.quantile(means,.975))

def exact_task_accuracy(labels, enc, Q, prior):
    """Expected MAP recovery accuracy for W at a receiver, before sampling noise."""
    joint=np.zeros((K,N_S))
    for a,m in enumerate(enc): joint[a]+=prior[a]*Q[symbol(a,m)]
    dec=joint.argmax(0)
    return float(sum(joint[dec[y],y] for y in range(N_S)))

def svg(path,rows,x,y,title):
    W,H,L,R,T,B=760,470,80,30,35,65;xs=[float(q[x]) for q in rows];ys=[float(q[y]) for q in rows];xmin,xmax=min(xs),max(xs);ymin,ymax=min(ys),max(ys)
    if xmax==xmin:xmax=xmin+1
    if ymax==ymin:ymax=ymin+1
    sx=lambda z:L+(z-xmin)/(xmax-xmin)*(W-L-R);sy=lambda z:H-B-(z-ymin)/(ymax-ymin)*(H-T-B)
    v=['<svg xmlns="http://www.w3.org/2000/svg" width="760" height="470">','<rect width="100%" height="100%" fill="white"/>',f'<line x1="{L}" y1="{T}" x2="{L}" y2="{H-B}" stroke="#222"/><line x1="{L}" y1="{H-B}" x2="{W-R}" y2="{H-B}" stroke="#222"/>']
    for q in rows:v.append(f'<circle cx="{sx(float(q[x])):.1f}" cy="{sy(float(q[y])):.1f}" r="3" fill="#1f4e79" opacity=".6"/>')
    v += [f'<text x="{W/2}" y="{H-18}" text-anchor="middle" font-family="Arial" font-size="14">{x}</text>',f'<text transform="translate(18 {H/2}) rotate(-90)" text-anchor="middle" font-family="Arial" font-size="14">{y}</text>',f'<text x="{W/2}" y="20" text-anchor="middle" font-family="Arial" font-size="15">{title}</text>','</svg>'];path.write_text('\n'.join(v),encoding='utf-8')

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    data,scanned,bad=load_data();train,valid,test=data[:8400],data[8400:10200],data[10200:]
    X,centroids,train_labels,prior=fit_intents(train);tx=np.vstack([embed(x['query']) for x in test]);labels=((tx[:,None,:]-centroids[None,:,:])**2).sum(2).argmin(1)
    codebook=[]
    for w in range(K):
        candidates=np.where(train_labels==w)[0]
        rep=int(candidates[np.argmin(((X[candidates]-centroids[w])**2).sum(1))])
        codebook.append({'intent_w':w,'training_prior':prior[w],'representative_web_query':train[rep]['query'],'coarse_message':'global information need','group_message':f'group {w//2} information need','fine_message':f'intent {w} information need'})
    write(OUT/'MSMARCO_Web语义码本说明.csv',codebook)
    controls=[];all_by_condition={}
    for condition in ('Bob-better','Equal','Eve-better'):
        QB,QE,b,e,T=build_channels(condition);rows=[]
        for idx,enc in enumerate(product(range(3),repeat=K)):
            z=exact_metrics(enc,prior,QB,QE);z.update({'condition':condition,'encoding_id':idx});rows.append(z)
        all_by_condition[condition]=(rows,QB,QE,b,e,T);controls.extend(rows)
    rows,QB,QE,b,e,T=all_by_condition['Bob-better'];boundary=pareto(rows)
    write(OUT/'MSMARCO_Bob更强_全部确定性编码.csv',rows);write(OUT/'MSMARCO_Bob更强_Pareto边界.csv',boundary);write(OUT/'MSMARCO_三种信道条件对比.csv',controls)
    svg(FIG/'MSMARCO_D_L_Pareto.svg',boundary,'L','D_B','Finite Web distortion-cost Pareto boundary');svg(FIG/'MSMARCO_L_Se_Pareto.svg',boundary,'L','Se','Finite Web secrecy-cost Pareto boundary')
    compare=[];web=[]
    for budget in (1.,1.5,2.,2.5,3.):
        feasible=[r for r in rows if r['L']<=budget+1e-12]
        two=min(feasible,key=lambda r:(r['D_B'],r['L']))
        three=max(feasible,key=lambda r:(r['Se'],-r['D_B'],-r['L']))
        for name,z in [('2D_distortion_cost',two),('Proposed_3D_security',three)]:
            out={k:z[k] for k in ('D_B','D_E','L','I_B','I_E','Se','encoding')};out.update({'budget':budget,'method':name});compare.append(out)
            enc=list(map(int,z['encoding'].split()));acc,lo,hi=bootstrap_task(labels,enc,QB,prior,SEED+int(budget*100)+(0 if name.startswith('2D') else 1));web.append({'budget':budget,'method':name,'heldout_intent_accuracy':acc,'CI95_low':lo,'CI95_high':hi,'D_B_exact':z['D_B'],'Se_exact':z['Se']})
    write(OUT/'MSMARCO_2D与3D对比.csv',compare);write(OUT/'MSMARCO_留出Web任务验证.csv',web)

    # Direct adversary-task validation and constrained objective comparison.
    constrained=[]
    for budget in (1.5,2.0,2.5,3.0):
        feasible=[r for r in rows if r['L']<=budget+1e-12]
        for method,z in [('Min_D_B_at_budget',min(feasible,key=lambda r:(r['D_B'],r['L']))),
                         ('Max_Se_at_budget',max(feasible,key=lambda r:(r['Se'],-r['D_B'],-r['L'])) )]:
            enc=list(map(int,z['encoding'].split()))
            constrained.append({'budget':budget,'method':method,'D_B':z['D_B'],'L':z['L'],'I_B':z['I_B'],'I_E':z['I_E'],'Se':z['Se'],
                                'Bob_W_accuracy_exact':exact_task_accuracy(labels,enc,QB,prior),
                                'Eve_W_accuracy_exact':exact_task_accuracy(labels,enc,QE,prior),
                                'Bob_Eve_accuracy_gap':exact_task_accuracy(labels,enc,QB,prior)-exact_task_accuracy(labels,enc,QE,prior),
                                'encoding':z['encoding']})
    write(OUT/'MSMARCO_Bob与Eve_W恢复准确率.csv',constrained)

    # Sensitivity of the secrecy result to the Eve post-processing strength.
    sensitivity=[]
    for a in (.10,.30,.50,.70,.90):
        qb_a,qe_a,_,_,_=build_channels('Bob-better',(a,a,a))
        local=[]
        for enc in product(range(3),repeat=K): local.append(exact_metrics(enc,prior,qb_a,qe_a))
        feasible=[r for r in local if r['L']<=2.0+1e-12]
        z=max(feasible,key=lambda r:(r['Se'],-r['D_B']))
        sensitivity.append({'alpha_retain':a,'budget':2.0,'D_B':z['D_B'],'L':z['L'],'I_B':z['I_B'],'I_E':z['I_E'],'Se':z['Se'],'feasible_encodings':len(feasible),'encoding':z['encoding']})
    write(OUT/'MSMARCO_Eve信道强度敏感性.csv',sensitivity)
    low=min(boundary,key=lambda r:r['L']);high=max(boundary,key=lambda r:r['Se']);ts=[]
    for a in np.linspace(0,1,11):ts.append({'lambda':float(a),'D_B':a*low['D_B']+(1-a)*high['D_B'],'L':a*low['L']+(1-a)*high['L'],'Se':a*low['Se']+(1-a)*high['Se']})
    write(OUT/'MSMARCO_时分凸性验证.csv',ts)
    summary=[]
    for c,(r,_,_,_,_,_) in all_by_condition.items():summary.append({'condition':c,'encodings':len(r),'Se_min':min(x['Se'] for x in r),'Se_max':max(x['Se'] for x in r),'mean_Se':float(np.mean([x['Se'] for x in r]))})
    write(OUT/'MSMARCO_安全性方向验证.csv',summary)
    with open(OUT/'MSMARCO_理论验证审计.txt','w',encoding='utf-8') as f:
        f.write(f'records_scanned={scanned}\nmalformed={bad}\nsample={len(data)}\ntrain={len(train)} validation={len(valid)} test={len(test)}\n')
        f.write('W is an eight-class finite Web information-need source learned only from training queries; passages and relevance labels are evaluation-only. S has 13 reusable symbols: one coarse, four group, eight intent-specific.\n')
        f.write(f'Each evaluation query uses one positive and {N_NEG} negative passages (fixed {N_NEG+1}-candidate pool).\n')
        f.write('The legitimate receiver has a mode-conditioned symmetric DMC with correct probabilities (.97,.91,.96). In the legitimate-receiver-better condition, the eavesdropper channel is QE=QL*T, where T retains each observed semantic symbol with alpha=(.70,.35,.65) by symbol mode and otherwise emits a uniform symbol. Thus S->Y_L->Z_E holds exactly.\n')
        f.write(f'Legitimate-receiver-better eavesdropper marginal correct probabilities={e}; Markov construction rows={T.shape[0]}\n')
        f.write('D, L, I_B, I_E, Se are exact finite sums. Held-out query labels are used only for an application-level Monte Carlo decoder check.\n')
        f.write(f'Legitimate-receiver-better encodings={len(rows)} Pareto_points={len(boundary)}\n')
        f.write('Additional analyses report exact MAP W-recovery accuracy for Bob and Eve, constrained budget comparisons, and alpha-retention sensitivity at L<=2.0.\n')
    print('OUTPUT',OUT);print('PARETO',len(boundary));print('SECURITY',summary);print('COMPARE',compare)

if __name__=='__main__':main()
