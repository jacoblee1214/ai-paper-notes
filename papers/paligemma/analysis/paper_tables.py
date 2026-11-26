"""논문 PaliGemma(arXiv:2407.07726 v2)의 표를 옮겨 적은 데이터. 모든 값은 paper/paper.txt와 PDF 쪽에서 읽은 것이다.

각 항목 옆의 위치(표·절·부록)가 근거다. 그림에서 읽은 값은 figure_extract.py가 PDF 벡터에서 뽑아 figure_values.json에 둔다.
"""

# 표 1 (p.7). (구역, 과제, 제로샷 여부(⌞), 224px, 448px, 896px), 각 칸은 (값, 표준편차) 또는 None.
# 표준편차 = 같은 최적 하이퍼파라미터로 5번 다시 전이한 값의 표준편차(§6.1). 서버 제출 과제는 검증 분할에서 낸 값(표 1 캡션).
# 값 자체는 '5번 중 무작위 1번'(캡션)이다. MMVP, POPE, Objaverse Multiview는 표준편차가 없다.
TABLE1 = [
    ('Image captioning', 'COCOcap', False, (141.9, 0.3), (144.6, 0.5), None),
    ('Image captioning', 'NoCaps', True, (121.7, 0.3), (123.6, 0.7), None),
    ('Image captioning', 'COCO-35L (en)', False, (139.2, 0.4), (141.2, 0.6), None),
    ('Image captioning', 'COCO-35L (avg34)', False, (113.7, 0.3), (115.8, 0.1), None),
    ('Image captioning', 'XM3600 (en)', True, (78.0, 0.8), (80.0, 0.3), None),
    ('Image captioning', 'XM3600 (avg35)', True, (41.9, 0.0), (42.4, 0.1), None),
    ('Image captioning', 'Screen2Words', False, (117.6, 0.7), (119.6, 0.7), None),
    ('Image captioning', 'TextCaps', False, (127.5, 1.0), (153.9, 0.5), None),
    ('Image captioning', 'SciCap', False, (162.3, 0.7), (181.5, 0.6), None),
    ('Image captioning', 'WidgetCap', False, (136.1, 1.4), (148.4, 0.3), None),
    ('Visual question answering', 'VQAv2', False, (83.2, 0.4), (85.6, 0.2), None),
    ('Visual question answering', 'MMVP', True, (47.3, None), (45.3, None), None),
    ('Visual question answering', 'POPE', True, (86.0, None), (87.0, None), None),
    ('Visual question answering', 'Objaverse Multiview', True, (62.7, None), (62.8, None), None),
    ('Visual question answering', 'OKVQA', False, (63.5, 0.3), (63.2, 0.2), None),
    ('Visual question answering', 'AOKVQA-MC', False, (76.4, 0.4), (76.9, 0.3), None),
    ('Visual question answering', 'AOKVQA-DA', False, (61.9, 0.6), (63.2, 0.5), None),
    ('Visual question answering', 'GQA', False, (65.6, 0.4), (67.0, 0.3), None),
    ('Visual question answering', 'xGQA (avg7)', True, (57.3, 0.2), (57.9, 0.5), None),
    ('Visual question answering', 'NLVR2', False, (90.0, 0.2), (88.9, 0.3), None),
    ('Visual question answering', 'MARVL (avg5)', True, (80.6, 0.3), (76.8, 0.3), None),
    ('Visual question answering', 'AI2D', False, (72.1, 0.7), (73.3, 0.7), None),
    ('Visual question answering', 'ScienceQA', False, (95.4, 0.3), (95.9, 0.2), None),
    ('Visual question answering', 'RSVQA-lr', False, (92.6, 0.3), (93.1, 0.7), None),
    ('Visual question answering', 'RSVQA-hr (test)', False, (92.6, 0.1), (92.8, 0.1), None),
    ('Visual question answering', 'RSVQA-hr (test2)', False, (90.6, 0.1), (90.5, 0.2), None),
    ('Visual question answering', 'ChartQA (human)', False, (40.0, 0.7), (54.2, 1.1), None),
    ('Visual question answering', 'ChartQA (aug)', False, (74.2, 0.2), (88.5, 0.3), None),
    ('Visual question answering', 'VizWizVQA', False, (73.7, 0.2), (75.5, 0.5), None),
    ('Visual question answering', 'TallyQA (simple)', False, (81.7, 0.2), (84.9, 0.1), None),
    ('Visual question answering', 'TallyQA (complex)', False, (69.6, 0.1), (72.3, 0.2), None),
    ('Visual question answering', 'CountBenchQA', False, (81.9, 1.6), (83.1, 2.1), None),
    ('Visual question answering', 'OCR-VQA', False, (72.3, 0.4), (74.6, 0.4), (74.9, None)),
    ('Visual question answering', 'TextVQA', False, (55.5, 0.2), (73.2, 0.2), (76.5, None)),
    ('Visual question answering', 'DocVQA', False, (43.7, 0.5), (78.0, 0.3), (84.8, None)),
    ('Visual question answering', 'InfoVQA', False, (28.5, 0.4), (40.5, 0.3), (47.8, None)),
    ('Visual question answering', 'ST-VQA', False, (63.3, 0.5), (81.8, 0.3), (84.4, None)),
    ('Image segmentation', 'RefCOCO (testA)', False, (75.7, 0.1), (77.9, 0.1), (78.7, None)),
    ('Image segmentation', 'RefCOCO (testB)', False, (70.7, 0.2), (72.4, 0.2), (73.9, None)),
    ('Image segmentation', 'RefCOCO+ (testA)', False, (71.9, 0.1), (74.2, 0.2), (76.1, None)),
    ('Image segmentation', 'RefCOCO+ (testB)', False, (64.5, 0.6), (64.5, 0.2), (66.9, None)),
    ('Image segmentation', 'RefCOCOg (test)', False, (68.2, 0.1), (71.0, 0.1), (72.7, None)),
    ('Video input', 'ActivityNet-QA', False, (50.8, 0.4), None, None),
    ('Video input', 'ActivityNet-CAP', False, (34.6, 0.6), None, None),
    ('Video input', 'MSRVTT-QA', False, (50.1, 0.1), None, None),
    ('Video input', 'MSRVTT-CAP', False, (70.5, 0.9), None, None),
    ('Video input', 'MSVD-QA', False, (60.2, 0.3), None, None),
    ('Video input', 'VATEX', False, (79.7, 0.4), None, None),
]

# 표 1에서 음영으로 표시된 '해상도 민감' 9개 과제(p.7 이미지, 그림 9의 9 tasks와 일치)
RES_SENSITIVE = ['TextCaps', 'SciCap', 'ChartQA (human)', 'ChartQA (aug)', 'OCR-VQA', 'TextVQA', 'DocVQA', 'InfoVQA', 'ST-VQA']

# 표 2 (p.13): 모든 과제에 하나의 단순 설정(lr 1e-5, bs 256, 드롭아웃·라벨 스무딩·가중치 감쇠 없음, 에폭만 과제별 최대 10)을 썼을 때의 상대 후회
TABLE2 = {'[None,2.5%)': 37, '[2.5%,5%)': 2, '[5%,10%)': 2, '[10%,100%]': 2}
TABLE2_TASKS = {'ChartQA (human)': 3.2, 'RefCOCO (val)': 4.8, 'RefCOCOg (val)': 5.8, 'ScienceQA': 6.7, 'RefCOCO+ (val)': 10.7, 'SciCap': 60.5}

# 표 4 (부록 H, p.46): (지표, 표 1, 단일 단순, 다중 접두, 다중 무접두). 표 1 열은 검증 분할 값인 항목이 있다(DocVQA val 등).
TABLE4 = [
    ('AI2D', 72.1, 73.1, 72.3, 72.3),
    ('COCOcap', 141.9, 141.7, 139.5, 138.9),
    ('NoCaps', 121.7, 121.6, 118.4, 98.9),
    ('DocVQA (val)', 37.8, 38.6, 32.2, 31.3),
    ('GQA', 65.6, 65.4, 64.9, 64.2),
    ('xGQA (avg7)', 57.3, 56.9, 55.8, 52.6),
    ('InfoVQA (val)', 25.5, 25.0, 20.8, 21.4),
    ('OCR-VQA', 72.3, 73.3, 71.4, 71.4),
    ('OKVQA', 63.5, 63.1, 67.0, 58.9),
    ('RefCOCO (val)', 73.4, 69.9, 68.4, 68.3),
    ('RefCOCO+ (val)', 68.3, 61.0, 61.2, 60.9),
    ('RefCOCOg (val)', 67.7, 63.7, 61.8, 61.8),
    ('RSVQA-hr (test)', 92.6, 92.7, 92.1, 92.2),
    ('RSVQA-hr (test2)', 90.6, 90.8, 90.0, 90.2),
    ('RSVQA-lr', 92.6, 93.7, 93.4, 93.0),
    ('SciCap', 162.3, 64.0, 74.7, 74.7),
    ('Screen2Words', 117.6, 114.7, 111.6, 110.6),
    ('ST-VQA (val)', 61.6, 62.0, 58.3, 57.8),
    ('TallyQA (simple)', 81.7, 81.3, 80.4, 80.7),
    ('TallyQA (complex)', 69.6, 69.6, 69.6, 69.9),
    ('CountBenchQA', 81.9, 83.5, 80.6, 78.3),
    ('TextCaps', 127.5, 128.3, 122.5, 122.3),
    ('TextVQA (val)', 59.0, 57.9, 56.0, 56.5),
    ('VQAv2 (minival)', 82.1, 81.9, 83.4, 82.9),
    ('VizWizVQA (val)', 76.4, 75.9, 75.7, 76.0),
    ('WidgetCap', 136.1, 136.4, 129.1, 131.5),
]
TABLE4_AVG = (84.6, 80.2, 78.9, 77.6)  # 논문이 적은 평균 행

# 부록 J(pp.48-49): (과제, 해상도, 에폭, 배치, 학습률, 가중치 감쇠, LLM 드롭아웃, 라벨 스무딩, ViT 동결, 디코딩)
HPARAMS = [
    ('COCOcap', 224, 5, 256, 1e-05, 1e-06, 0.0, 0.0, False, 'beam n=2'),
    ('COCOcap', 448, 5, 256, 1e-05, 1e-06, 0.0, 0.0, False, 'beam n=2'),
    ('COCO-35L', 224, 25, 256, 1e-05, 1e-06, 0.0, 0.0, False, 'greedy'),
    ('COCO-35L', 448, 25, 256, 1e-05, 1e-06, 0.0, 0.0, False, 'greedy'),
    ('Screen2Words', 224, 10, 256, 1e-05, 0.0, 0.3, 0.2, False, 'greedy'),
    ('Screen2Words', 448, 10, 256, 1e-05, 0.0, 0.3, 0.2, False, 'greedy'),
    ('TextCaps', 224, 5, 256, 1e-05, 1e-06, 0.0, 0.0, False, 'beam n=3'),
    ('TextCaps', 448, 5, 256, 1e-05, 1e-06, 0.0, 0.0, False, 'beam n=3'),
    ('SciCap', 224, 80, 256, 3e-05, 0.0, 0.1, 0.1, False, 'greedy'),
    ('SciCap', 448, 80, 256, 3e-05, 0.0, 0.1, 0.1, False, 'greedy'),
    ('WidgetCap', 224, 4, 64, 3e-06, 3e-07, 0.1, 0.1, False, 'greedy'),
    ('WidgetCap', 448, 4, 64, 3e-06, 3e-07, 0.1, 0.1, False, 'greedy'),
    ('VQAv2', 224, 10, 256, 1e-05, 1e-06, 0.0, 0.0, False, 'greedy'),
    ('VQAv2', 448, 10, 256, 1e-05, 0.0, 0.0, 0.0, False, 'greedy'),
    ('OKVQA', 224, 10, 128, 5e-06, 0.0, 0.0, 0.0, False, 'greedy'),
    ('OKVQA', 448, 10, 128, 5e-06, 0.0, 0.0, 0.0, False, 'greedy'),
    ('AOKVQA-MC', 224, 15, 128, 5e-06, 0.0, 0.0, 0.0, False, 'greedy'),
    ('AOKVQA-MC', 448, 15, 128, 5e-06, 0.0, 0.0, 0.0, False, 'greedy'),
    ('AOKVQA-DA', 224, 10, 128, 5e-06, 0.0, 0.0, 0.0, False, 'greedy'),
    ('AOKVQA-DA', 448, 10, 128, 5e-06, 0.0, 0.0, 0.0, False, 'greedy'),
    ('GQA', 224, 1, 256, 1e-05, 0.0, 0.0, 0.0, False, 'greedy'),
    ('GQA', 448, 1, 256, 1e-05, 0.0, 0.0, 0.0, True, 'greedy'),
    ('NLVR2', 224, 3, 256, 1e-05, 1e-06, 0.0, 0.0, False, 'greedy'),
    ('NLVR2', 448, 10, 256, 3e-06, 3e-07, 0.0, 0.0, False, 'greedy'),
    ('AI2D', 224, 10, 256, 1e-05, 1e-06, 0.0, 0.0, False, 'greedy'),
    ('AI2D', 448, 10, 256, 1e-05, 1e-06, 0.0, 0.0, False, 'greedy'),
    ('ScienceQA', 224, 20, 128, 1e-05, 0.0, 0.0, 0.0, True, 'greedy'),
    ('ScienceQA', 448, 20, 128, 1e-05, 0.0, 0.0, 0.0, True, 'greedy'),
    ('RSVQA-lr', 224, 3, 256, 3e-06, 0.0, 0.0, 0.2, False, 'greedy'),
    ('RSVQA-lr', 448, 3, 256, 3e-06, 0.0, 0.0, 0.2, False, 'greedy'),
    ('RSVQA-hr', 224, 1, 256, 1e-05, 0.0, 0.0, 0.0, False, 'greedy'),
    ('RSVQA-hr', 448, 1, 256, 1e-05, 0.0, 0.0, 0.0, False, 'greedy'),
    ('ChartQA', 224, 30, 256, 1e-05, 1e-06, 0.1, 0.2, False, 'greedy'),
    ('ChartQA', 448, 30, 256, 1e-05, 1e-06, 0.1, 0.2, False, 'greedy'),
    ('VizWizVQA', 224, 10, 256, 1e-05, 0.0, 0.0, 0.0, False, 'greedy'),
    ('VizWizVQA', 448, 10, 256, 1e-05, 0.0, 0.0, 0.0, False, 'greedy'),
    ('TallyQA', 224, 2, 256, 1e-05, 0.0, 0.0, 0.0, False, 'greedy'),
    ('TallyQA', 448, 2, 256, 1e-05, 1e-06, 0.0, 0.0, False, 'greedy'),
    ('OCR-VQA', 224, 3, 128, 3e-06, 0.0, 0.0, 0.0, False, 'greedy'),
    ('OCR-VQA', 448, 3, 128, 3e-06, 0.0, 0.0, 0.0, False, 'greedy'),
    ('OCR-VQA', 896, 3, 128, 1e-05, 0.0, 0.0, 0.0, False, 'greedy'),
    ('TextVQA', 224, 5, 256, 3e-06, 0.0, 0.0, 0.0, False, 'greedy'),
    ('TextVQA', 448, 10, 256, 3e-06, 3e-07, 0.0, 0.0, False, 'greedy'),
    ('TextVQA', 896, 10, 256, 3e-06, 0.0, 0.0, 0.0, False, 'greedy'),
    ('DocVQA', 224, 10, 256, 1e-05, 1e-06, 0.0, 0.0, False, 'greedy'),
    ('DocVQA', 448, 10, 256, 1e-05, 1e-06, 0.0, 0.0, False, 'greedy'),
    ('DocVQA', 896, 10, 256, 1e-05, 1e-06, 0.0, 0.0, False, 'greedy'),
    ('InfoVQA', 224, 3, 256, 1e-05, 1e-06, 0.0, 0.4, False, 'greedy'),
    ('InfoVQA', 448, 3, 128, 1e-05, 1e-06, 0.0, 0.4, False, 'greedy'),
    ('InfoVQA', 896, 3, 32, 3e-06, 3e-07, 0.0, 0.4, False, 'greedy'),
    ('ST-VQA', 224, 3, 256, 1e-05, 1e-06, 0.0, 0.1, False, 'greedy'),
    ('ST-VQA', 448, 3, 128, 1e-05, 1e-06, 0.0, 0.1, False, 'greedy'),
    ('ST-VQA', 896, 3, 32, 3e-06, 3e-07, 0.0, 0.1, False, 'greedy'),
    ('RefCOCO', 224, 100, 256, 3e-05, 0.0, 0.1, 0.3, False, 'greedy'),
    ('RefCOCO', 448, 100, 256, 1e-05, 0.0, 0.0, 0.3, False, 'greedy'),
    ('RefCOCO', 896, 100, 64, 1e-05, 0.0, 0.0, 0.3, False, 'greedy'),
    ('ActivityNet-QA', 224, 1, 128, 1e-05, 1e-06, 0.0, 0.0, False, 'greedy'),
    ('ActivityNet-CAP', 224, 1, 128, 1e-05, 1e-06, 0.0, 0.0, True, 'greedy'),
    ('MSRVTT-QA', 224, 1, 128, 1e-05, 0.0, 0.0, 0.0, True, 'greedy'),
    ('MSRVTT-CAP', 224, 20, 128, 1e-05, 0.0, 0.0, 0.0, True, 'greedy'),
    ('MSVD-QA', 224, 1, 128, 3e-06, 3e-07, 0.0, 0.0, False, 'greedy'),
    ('VATEX', 224, 10, 128, 3e-06, 3e-07, 0.0, 0.0, False, 'greedy'),
]

# 본문 보고값
STAGE1 = {'examples': 1_000_000_000, 'res': 224, 'n_img': 256, 'n_txt': 128, 'reported_tokens': 350e9,
          'reported_days_lt': 3, 'tok_per_s_per_dev': 5189, 'devices': 256, 'mfu_reported': 0.55}   # §3.2.2, §3.2.6
STAGE2 = [{'res': 448, 'examples': 50_000_000, 'n_img': 1024, 'n_txt': 512, 'hours': 15},
          {'res': 896, 'examples': 10_000_000, 'n_img': 4096, 'n_txt': 512, 'hours': 15}]  # §3.2.3, §3.2.6
STAGE2_REPORTED_TOKENS = 90e9   # 'both Stage2 combined about 90 B tokens' (§3.2.6)

# 논문 본문에서 확인한 평가 세트 크기. 논문은 시험 세트 크기를 거의 적지 않는다(부록 B는 학습 예제 수만).
#   CountBenchQA 540장 중 이미지가 남은 것은 490(부록 D). MMVP는 150쌍 300문항, 우연 수준 25%(arXiv:2401.06209 §2, Table 1, 우리가 확인).
# 아래 TEST_N은 '논문에 없고 공개 벤치마크의 통상 크기로 아는 값'이다. 공식 페이지로 확인하지 않았다(외부 지식, 근사).
TEST_N_IN_PAPER_OR_CHECKED = {'CountBenchQA': 540, 'MMVP (pairs)': 150}
TEST_N_RECALLED_UNVERIFIED = {'ChartQA (human)': 1250, 'ChartQA (aug)': 1250, 'DocVQA': 5188, 'InfoVQA': 3288,
                              'TextVQA': 5734, 'AI2D': 3088, 'ScienceQA': 2017, 'POPE': 9000}
