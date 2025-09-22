# =======================
# C:\vpdetect\test_score.ps1
# =======================

# 0) 콘솔 출력 UTF-8 고정(한글 깨짐 방지)
try {
    $OutputEncoding = [Console]::OutputEncoding = [Text.UTF8Encoding]::UTF8
    chcp 65001 | Out-Null
} catch {}

Set-StrictMode -Version Latest

# 1) 서버 기본 설정
$BASE = "http://localhost:3001"
$PATH_SCORE = "/score"
$PATH_EXPLAIN = "/debug/score_explain"

# 사용할 프로필
$profileName = "ml_boosted"

# 2) 공용 함수: JSON을 UTF-8로 POST (ij 불필요)
function Send-Json {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [hashtable] $BodyObject,
        [Parameter(Mandatory)]
        [string] $Path
    )
    $json  = $BodyObject | ConvertTo-Json -Depth 10
    $bytes = [System.Text.Encoding]::UTF8.GetBytes($json)
    $ms    = [System.IO.MemoryStream]::new($bytes)
    $uri   = "{0}{1}" -f $BASE, $Path
    return Invoke-RestMethod -Uri $uri -Method Post -ContentType "application/json; charset=utf-8" -Body $ms
}

# 3) 테스트 케이스 (Here-String으로 한글 안전)
$cases = @(
    [pscustomobject]@{
        name = "가족사칭-긴급-금전(HIGH 기대)"
        text = @"
엄마 휴대폰이 물에 빠져서 지금 안전계좌로 이체 좀 해줘. 인증번호도 필요해
"@
    },
    [pscustomobject]@{
        name = "검찰사칭-인증번호(CRITICAL 기대)"
        text = @"
서울중앙지검 수사관입니다. 계좌가 범죄에 연루되어 인증번호 확인이 필요합니다
"@
    },
    [pscustomobject]@{
        name = "재난지원금-단축링크(WARN~HIGH 기대)"
        text = @"
국가 재난지원금 신청 링크입니다: bit.ly/abcd1234 지금 신청하세요
"@
    },
    [pscustomobject]@{
        name = "관세미납-앱설치(WARN~HIGH 기대)"
        text = @"
택배 관세 미납으로 반송 예정입니다. 확인을 위해 앱 설치 후 로그인하세요
"@
    },
    [pscustomobject]@{
        name = "대환대출(WARN~HIGH 기대)"
        text = @"
은행 대환대출 금리 우대, 오늘만 가능하니 바로 신청하세요
"@
    }
)

Write-Host ("=== Score 테스트 시작 (profile={0}) ===" -f $profileName) -ForegroundColor Cyan

# 4) 점수 테스트
foreach ($c in $cases) {
    $body = @{ text = $c.text; profile = $profileName }
    try {
        $res   = Send-Json -BodyObject $body -Path $PATH_SCORE
        $score = "{0:N3}" -f $res.score
        $level = $res.level
        Write-Host ("[{0}] score={1} level={2}" -f $c.name, $score, $level)
    }
    catch {
        Write-Host ("[{0}] 호출 실패: {1}" -f $c.name, $_.Exception.Message) -ForegroundColor Red
    }
}

Write-Host "=== 완료 ===" -ForegroundColor Cyan

# 5) 상세 근거(첫 케이스)
try {
    Write-Host "`n[상세 근거] 첫 케이스 ----------------" -ForegroundColor Yellow
    $firstBody = @{ text = $cases[0].text; profile = $profileName }
    $detail    = Send-Json -BodyObject $firstBody -Path $PATH_EXPLAIN
    $detail | ConvertTo-Json -Depth 10
}
catch {
    Write-Host ("상세 근거 조회 실패: {0}" -f $_.Exception.Message) -ForegroundColor Red
}
