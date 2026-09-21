"""构造"纯成功轨迹"的 fixed 集。

三条约束：
 1. 只要 success=True（done 且 reward>=100）
 2. 按任务类型分层，避免像旧 fixed 那样 25 条全是"测熔点"
 3. 来源在 grpo / pf / te 三族间轮转均衡 —— 若全取自某一族，
    该族就是在自己的输出上被打分，等于把 own 的主场优势搬进 fixed
"""
import json, glob, os, re, shutil, collections, random
random.seed(0)
ROOT="runs/sciworld_eval"
SRC={"grpo":"sciworld_grpo_qwen7b_BASELINE_nopf_20260913_102029",
     "pf":"sciworld_grpo_qwen7b_planK3_nognorm_v4_20260906_091404",
     "te":"sciworld_TEabl_A_20260916_190725"}
STEPS=(50,100,150,200,250)
def ttype(t): return re.sub(r'\d+','N',(t or "").strip())[:55]

# 收集：task_id -> [(fam, step, path, tasktype)]
pool=collections.defaultdict(list)
for fam,exp in SRC.items():
    for s in STEPS:
        for f in glob.glob(f"{ROOT}/{exp}/step{s}/sciworld_*.json"):
            try: j=json.load(open(f))
            except Exception: continue
            if not j.get("success"): continue
            tid=int(re.search(r"sciworld_(\d+)",f).group(1))
            pool[tid].append((fam,s,f,ttype(j.get("task_description"))))
print(f"有成功轨迹的 task: {len(pool)}")

by_type=collections.defaultdict(list)
for tid,v in pool.items(): by_type[v[0][3]].append(tid)
print(f"任务类型数: {len(by_type)}")

TARGET=60
# 分层配额：每类按占比分，至少 1
tot=sum(len(v) for v in by_type.values())
quota={t: max(1, round(TARGET*len(v)/tot)) for t,v in by_type.items()}
fam_count=collections.Counter()
picked=[]
for t,tids in sorted(by_type.items(), key=lambda x:-len(x[1])):
    for tid in sorted(tids)[:quota[t]]:
        # 在该 task 的所有成功轨迹里，选目前贡献最少的那个族
        cands=pool[tid]
        cands.sort(key=lambda c:(fam_count[c[0]], c[1]))
        fam,s,f,_=cands[0]
        fam_count[fam]+=1
        picked.append((tid,fam,s,f))
    if len(picked)>=TARGET: break
picked=picked[:TARGET]
picked.sort(key=lambda x:x[0])

OUT="runs/sciworld_eval/FIXED_SUCCESS60"
shutil.rmtree(OUT, ignore_errors=True); os.makedirs(OUT)
meta=[]
for tid,fam,s,f in picked:
    shutil.copy(f, f"{OUT}/sciworld_{tid}.json")
    meta.append({"task_id":tid,"src_family":fam,"src_step":s,"src":f})
json.dump(meta, open(f"{OUT}/_manifest.json","w"), ensure_ascii=False, indent=1)
print(f"\n已写入 {OUT}: {len(picked)} 条")
print("来源族分布:", dict(fam_count))
print("任务类型分布:")
for k,v in collections.Counter(ttype(json.load(open(f"{OUT}/sciworld_{t}.json")).get("task_description")) for t,_,_,_ in picked).most_common():
    print(f"  {v:3d}  {k}")
# 复核
ok=sum(1 for t,_,_,_ in picked if json.load(open(f"{OUT}/sciworld_{t}.json")).get("success"))
print(f"\n复核: success=True 的有 {ok}/{len(picked)}")
