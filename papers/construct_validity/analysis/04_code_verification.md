# 04. 논문 대 공개 자료·코드 정적 대조 (D1–D10)

- 대상: GitHub `am-bean/benchmark_review`(HEAD `ac85a25`, 마지막 push 2025-09-29, API 크기 51.6 MB, 라이선스 파일 없음, 별 2)와 HF `ambean/construct-validity-review`(커밋 `312b4f8`, CC BY-SA 4.0).
- 저장소는 **분석 노트북 3개(`systematic_search`, `coding_results`, `citations_search`) + 평가자 일치도 노트북 1개 + 데이터 CSV**입니다. 노트북은 **읽기만** 했고 실행하지 않았습니다(OpenAI 호출과 PMLR 다운로드가 들어 있음). 내려받은 것: 필요한 파일만 67 MB(HF 4.3 MB + GitHub 일부), 디스크 합계 71 MB(`du -sh data`). 임베딩 CSV(78 MB)는 받지 않았습니다.
- 재계산은 우리가 새로 짠 스크립트(`03`)로 했고 노트북 결과는 저장된 출력만 읽었습니다.

## D1. 비율의 분모가 455행입니다 (논문은 445편)

`coding_results.ipynb` 셀 47–48: `included_df.shape = (455, 130)`, `bibkey.nunique() = 445`. 셀 54–55의 비율(`quick_percents`)은 455행 위에서 계산하고, 저장된 출력은 phenomenon_defined Yes 348(0.7648), 논쟁적 225(0.4945)입니다. **논문이 보고한 78.2%, 47.8%와 노트북 자신의 출력이 다릅니다.** 논문 값은 348/445, 그림 3은 분자(455행)와 분모(445)가 섞인 것으로 재현됩니다(`03` §2.1). 영향은 작지만(평균 0.36%p) 중복 10행의 코딩이 서로 다릅니다.

## D2. 정규화(`_clean`)가 응답을 후하게 합칩니다

노트북 셀 20–32의 `*_map`: 타당성 논의는 "Somewhat", "implicitly", "indirectly address it", "partially" 등을 모두 Yes로, 현실 비교는 "No - but …scraped from social media"를 Realistic으로 바꿉니다. 원본 'Yes'만 세면 223/455(49.0%)로 53.4%와 다릅니다(`03` §2.2). 코드북 문서(부록 B)는 이 매핑 규칙을 적지 않았습니다.

## D3. 선별 단계 수치가 노트북 출력과 최종 파일에서 다릅니다

`systematic_search.ipynb` 셀 42·45·49의 저장된 출력은 inclusion Yes 896, Language 670 등이고, 이후 `included_papers_r2.csv`·`included_papers_embeddings.csv`(셀 56–58)는 938, 709입니다. 즉 LLM 선별이 한 번 이상 다시 돌았고 최종 값이 논문 표 7과 일치하는 파일은 r2입니다(`03` §1.1). 재실행 시점·판본(gpt-4o-mini, temperature 0)은 기록되지 않았습니다. 또 셀 46은 contribution 질의를 Multimodal에만 돌리는 코드인데 최종 파일의 contribution 라벨은 Language 행에만 있어 노트북에 보이지 않는 실행이 더 있었습니다. 522 → 최종 목록에서 Language/Benchmark 450 → 430, Multimodal 137 → 92 감소의 원인(`03` §1.1)은 공개 코드로 설명되지 않습니다.

## D4. 키워드 규칙은 논문 서술보다 구체적입니다

셀 9·14·19·25: `'LLM' in x`(대소문자 구분) 또는 `'language model' in x.lower()`, 그리고 `'benchmark' in x.lower()`를 제목 또는 초록에서 찾음. ACL 계열은 초록과 doi가 있는 논문만, 데모·산업·학생 워크숍 트랙은 제외(셀 29). 논문 §3은 이 세부를 쓰지 않았습니다. 이 규칙을 그대로 적용하면 ICML 외 모든 학회가 목록과 거의 일치합니다(`03` §1.2). ICML은 셀 8이 실행 시 PMLR RSS에서 받아 저장소에 원본이 없습니다.

## D5. 평가자 일치도: 논문은 BPK만 보고하고 BPK 코드는 저장소에 없습니다

`Annotator agreement.ipynb`는 Cohen κ(평균 0.191), Krippendorff α(0.173), Gwet AC1(0.191), 일치율(0.683)을 계산해 출력합니다. 저장소 전체에서 BPK(Brennan–Prediger) 구현은 찾지 못했습니다(세 노트북 grep: BPK 계열 문자열이 coding_results의 1건뿐). 논문이 보고한 BPK 표 9는 이 코드로 만든 것이 아닙니다. 우리가 같은 정의로 41쌍에서 재계산한 평균 BPK는 0.502이고 Cohen κ는 0.147입니다(`03` §3). 노트북의 일치율 0.683은 논문의 68.11%와 같은 수준입니다.

## D6. 이중 코딩 규모

노트북은 `merged_annotations.csv` 82행(41쌍)을 쓰고 논문은 46편입니다. 코드북의 중복 bibkey 9개는 이 41쌍과 겹치지 않습니다(`03` §3.1).

## D7. 공개 자료 간 일치

- HF 코드북과 GitHub `data/LLM_benchmark_codebook.csv`: 둘 다 455행, bibkey 순서 동일. HF는 `new_taxonomy_root`, `new_taxonomy_leaf` 두 열이 더 있습니다(72열 대 70열).
- `final_list.csv` 522행과 `included_papers_r2.csv`는 제목으로 전부 연결됩니다(522/522).
- 코드북 검토자 29명, 주 검토자 열의 합 454(빈칸 1).

## D8. 개인정보

GitHub `data/coding_responses.csv`에는 검토자 **이메일 열**이 있습니다. 받았다가 즉시 지웠고 쓰지 않았습니다. 게시본에는 어떤 원자료도 넣지 않습니다.

## D9. 라이선스

HF는 CC BY-SA 4.0, GitHub 저장소는 라이선스 없음(코드도 데이터도 재사용 권리가 불분명). 따라서 게시본에는 원자료 사본을 넣지 않고 우리 스크립트와 요약 JSON만 둡니다.

## D10. 재현 제안 (실행하지 않음)

| 번호 | 무엇 | 필요 | 비고 |
|---|---|---|---|
| 1 | ICML 목록을 PMLR RSS에서 받아 46,114 합과 ICML 171편 재집계 | 네트워크, 수 MB | 승인 불필요 수준이나 이번에는 하지 않음 |
| 2 | 선별 1단계를 gpt-4o-mini로 재실행해 표 8의 TI/FI 재현 | OpenAI API(비용·외부 전송) | **승인 필요**. 2,189편 약 0.08 USD(노트북 추정) |
| 3 | 두 번째 검토자 일치도를 새 표본으로 다시 측정 | 사람 또는 LLM 평가자 | LLM이면 API 승인 필요 |
