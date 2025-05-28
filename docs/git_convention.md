# mlops

## 1. Branch 전략

### 1.1 Branch 종류

### 1. **`main` 브랜치**
   - **라벨**: `main`
   - **설명**: `main` 브랜치는 항상 배포 가능한 상태여야 하므로, 이 브랜치는 최종적인 배포용 코드를 포함합니다.
   - **작업**: `main` 브랜치에는 직접 작업하지 않고, 항상 다른 브랜치에서 merge된 후 최신 상태를 유지합니다.

### 2. **`develop` 브랜치**
   - **라벨**: `development`
   - **설명**: `develop` 브랜치는 새로운 기능들이 통합되는 브랜치입니다. 이곳에서 모든 기능들이 합쳐지며, 코드가 테스트를 거쳐 배포 준비가 됩니다.
   - **작업**: `develop` 브랜치는 기능 개발이 완료되면 `feature` 브랜치나 `hotfix` 브랜치에서 merge됩니다.

### 3. **`feature/*` 브랜치**
   - **라벨**: `feature`
   - **설명**: 새로운 기능 개발을 위한 브랜치입니다. 각각의 기능은 별도의 `feature` 브랜치에서 작업하며, 해당 브랜치에서 작업이 완료되면 `develop` 브랜치로 merge됩니다.
   - **작업**: 각 기능을 개발하고, 완료되면 `develop` 브랜치에 merge하여 `develop`의 최신 상태에 반영합니다.

### 4. **`release/*` 브랜치**
   - **라벨**: `release`
   - **설명**: 배포를 준비하는 브랜치로, `develop` 브랜치의 최신 상태에서 분기하여 최종적인 배포 준비를 합니다. 버그 수정, 문서화, 패치 등을 여기서 처리합니다.
   - **작업**: `release` 브랜치에서 작업이 완료되면, `main` 브랜치로 merge하고, 태그를 추가하여 배포합니다.

### 흐름도 
`feature/*` → `develop` → `release/*` → `main`


### 1.2 예시

| 브랜치 유형  | 목적                                | 네이밍 규칙            | 예시                         |
|-------------|-------------------------------------|------------------------|-----------------------------|
| **main**    | 배포 가능한 안정적인 코드 관리      | `main`                 | `main`                      |
| **develop** | 개발 브랜치, 새로운 기능 통합 및 테스트 | `develop`              | `develop`                   |
| **feature** | 기능 개발 브랜치                    | `feature/기능명`        | `feature/login-page`        |
| **bugfix**  | 버그 수정 브랜치                    | `bugfix/버그내용`       | `bugfix/header-layout`      |
| **hotfix**  | 긴급 수정 브랜치 (배포 후 문제 해결) | `hotfix/수정내용`       | `hotfix/critical-error`     |
| **release** | 배포 준비 브랜치                    | `release/버전명`        | `release/v1.0.0`           |

> `main` 브랜치와 `develop` 브랜치는 Pull Request를 통해 배포됩니다.  
> push 를 통해서 merge 하지 않습니다.





## 2. 커밋 메시지 규칙

### 2.1 메시지 구조

```
<타입>: <제목>
<본문>
<푸터>
```

- **타입**: 메시지의 종류 (예: `feat`, `fix`, `docs` 등)
- **제목**: 간단한 변경 사항 설명
- **본문**: 상세한 설명 (옵션)
- **푸터**: 관련된 이슈 번호나 추가 정보 (옵션)

---

### 2.2 타입(Type)

| 타입     | 설명                                        |
|----------|---------------------------------------------|
| **feat** | 새로운 기능 추가                            |
| **fix**  | 버그 수정                                   |
| **docs** | 문서 수정 (예: README, 주석 수정 등)         |
| **style**| 코드 스타일 변경 (포맷팅, 세미콜론 추가/삭제 등, 기능 변화 없음) |
| **refactor** | 코드 리팩토링 (성능 개선 또는 구조 개선, 기능 변화 없음) |
| **test** | 테스트 코드 추가 및 수정                    |
| **chore** | 빌드 및 배포 설정, 패키지 관리 등 코드 외적인 작업 |
| **perf** | 성능 개선                                   |
| **ci**   | CI/CD 구성 변경                             |

## 기여 방법

1. 레포지토리를 포크(Fork)합니다.
2. 기능 브랜치를 생성합니다:  
   `git checkout -b feature-name`
3. 변경 사항을 커밋합니다:  
   `git commit -am 'Add new feature'`
4. 생성한 브랜치에 푸시합니다:  
   `git push origin feature-name`
5. Pull Request를 생성합니다.

## 당겨 올 때

1. upstream set-up
team repo를 upstream 이름으로 remote에 저장합니다. 

`git remote add upstream https://github.com/AIBootcamp13/mlops-cloud-project-mlops_1.git`

upstream 이름의 remote 저장소에는 push를 금지합니다.

`git remote set-url --push upstream no-push`

2. `git pull upstream main`
