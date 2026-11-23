"""논문 부록 D(Table 5-14)와 저장소 assessments/*.yaml을 파싱해 항목별 점수표를 만든다.
실행: python3 -I analysis/parse_assessments.py  (표준 라이브러리 + yaml)
"""
import re, json, pathlib, yaml
ROOT = pathlib.Path(__file__).resolve().parent.parent
txt = (ROOT/'paper/paper.txt').read_text().split('\n')
hdr = re.compile(r'Table (\d+): Assessment Report of ([\w\-]+)')
row = re.compile(r'^\s*([OTR])\.(\w+(?:\.\d+)?)\s+([01])\s+(.*)$')
paper = {}
cur = None
for ln in txt[1230:1668]:
    m = hdr.search(ln)
    if m:
        cur = m.group(2); paper.setdefault(cur, {}); continue
    m = row.match(ln)
    if m and cur:
        paper[cur][f'{m.group(1)}.{m.group(2)}'] = int(m.group(3))
name_map = {'SWE-Bench-Lancer':'swe-lancer','Bird-Bench':'bird-bench','CyBench':'cy-bench',
 'SWE-Bench-Verified':'swe-bench','tau-Bench':'tau-bench','MLE-Bench':'mle-bench',
 'WebArena':'webarena','GAIA':'gaia','OSWorld':'osworld','KernelBench':'kernel-bench'}
repo = {}
for k, v in name_map.items():
    d = yaml.safe_load((ROOT/f'repo/assessments/{v}.yaml').read_text().split('\n\n',1)[1])
    repo[k] = {i: int(x['score']) for i, x in d.items()}
out = {'paper': paper, 'repo': repo}
(ROOT/'.scratch').mkdir(exist_ok=True)
(ROOT/'.scratch/assess_raw.json').write_text(json.dumps(out, indent=1))
for k in paper:
    print(k, len(paper[k]), len(repo[k]), 'diff:', {i:(paper[k].get(i), repo[k].get(i)) for i in set(paper[k])|set(repo[k]) if paper[k].get(i)!=repo[k].get(i)})
