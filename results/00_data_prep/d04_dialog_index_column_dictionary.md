# d04_dialog_index.parquet 컬럼 설명

`scripts/build_index.py`(원본 인덱스 생성) + `scripts/clean_text.py`(텍스트 정제/r_gold 계산)로 만들어진 D04 dialog 단위 인덱스 파일의 컬럼 설명.

"출처" 컬럼: **JSON 원본** = 라벨링 json 파일의 값을 그대로 가져온 필드, **파생** = `build_index.py`/`clean_text.py`에서 코드로 계산·가공해서 새로 만든 필드.

| 컬럼명 | 출처 | 설명 |
|---|---|---|
| `call_id` | 파생 | `{j_code}_{s_code}`를 문자열로 조합해 생성한 통화 고유 ID (예: `J16_S000001`). json 파일 자체가 아니라 폴더/파일명에서 만듦. |
| `j_code` | 파생 | 라벨 json의 상위 폴더명에서 추출한 J코드 (예: J16~J19). json 내용이 아니라 경로에서 옴. |
| `s_code` | 파생 | 라벨 json 파일명(`S######.json`)에서 추출한 S코드. json 내용이 아니라 파일명에서 옴. |
| `category` | JSON 원본 | `dataSet.typeInfo.category` 값 그대로. |
| `subcategory` | JSON 원본 | `dataSet.typeInfo.subcategory` 값 그대로. |
| `speaker_id` | JSON 원본 | 각 발화(`dialogs[i]`)의 `speaker` 필드 값 그대로. |
| `speaker_type` | JSON 원본 | `typeInfo.speakers[].type` 값 그대로 (상담사1/2/3, 고객1/2/3). |
| `speaker_gender` | JSON 원본 | `typeInfo.speakers[].gender` 값 그대로. |
| `speaker_age` | JSON 원본 | `typeInfo.speakers[].age` 값 그대로. |
| `telephone_network` | JSON 원본 | `typeInfo.speakers[].telephone_network` 값 그대로 (본 데이터셋은 전부 `8k`). |
| `dialog_idx` | 파생 | json의 `dialogs` 배열을 순회하며 매긴 0-based 순번 (json에 명시적 필드로 존재하지 않음, 배열 순서로부터 계산). |
| `audioPath` | JSON 원본 | 각 발화의 `audioPath` 필드 값 그대로 (원천데이터 D04 폴더 기준 상대경로). |
| `duration` | JSON 원본 | 각 발화의 `duration` 필드 값 그대로(초). |
| `text` | JSON 원본 | 각 발화의 `text` 필드 값 그대로. disfluency 태그(`o/`, `n/`, `b/`, `l/` 등)와 교정쌍 표기 `(원문)/(교정)`, 불명확 발화 블록 `(( ))`가 원본 그대로 포함되어 있음. |
| `text_clean` | 파생 | `text`에서 disfluency 태그, 교정쌍(→교정 표기만 남김), 불명확 발화 블록 등을 제거해 정제한 텍스트 (`clean_text.py`에서 계산). |
| `disfluency_n_count` | 파생 | 원본 `text` 내 `n/` 태그 출현 횟수를 정규식으로 세어 계산. |
| `disfluency_o_count` | 파생 | 원본 `text` 내 `o/` 태그 출현 횟수를 정규식으로 세어 계산. |
| `disfluency_b_count` | 파생 | 원본 `text` 내 `b/` 태그 출현 횟수를 정규식으로 세어 계산. |
| `disfluency_l_count` | 파생 | 원본 `text` 내 `l/` 태그 출현 횟수를 정규식으로 세어 계산. |
| `disfluency_tag_count` | 파생 | 화이트리스트에 포함된 모든 disfluency 태그(o/n/b/l 및 소수 태그 i/p/f/h)의 총 출현 횟수 합산. |
| `correction_pair_count` | 파생 | `(원문)/(교정)` 형태의 교정쌍 출현 횟수를 정규식으로 세어 계산. |
| `unintelligible_flag` | 파생 | `True`면 `(( ))` 불명확 발화 블록이 있었고 그 결과 `text_clean`이 완전히 빈 문자열이 된 경우 (로직으로 판정). |
| `text_clean_too_short_flag` | 파생 | `text_clean` 길이가 2자 이하인 경우 `True` (로직으로 판정). |
| `syllable_count` | 파생 | `text_clean` 내 한글 음절 수를 정규식으로 세어 계산. |
| `r_gold` | 파생 | 발화속도 프록시 지표. `syllable_count / duration` (음절/초)로 계산. |
| `r_gold_valid_flag` | 파생 | 품질 플래그. `r_gold > 15`(비정상적으로 빠른 발화속도로, duration 대비 text 분량이 물리적으로 불가능한 수준)인 경우 `False`, 그 외 `True`로 계산. 제외된 케이스 상세는 `exclusion_log_r_gold_gt15.csv` 참조. |
