param([string]$OutputPath = "$PSScriptRoot/public_metadata.json")
$ErrorActionPreference = 'Stop'
$requests = @(
    @{label='WDS-XPINN'; uri='https://api.openalex.org/works/https://doi.org/10.1029/2023WR036641'},
    @{label='EMOC'; uri='https://api.openalex.org/works/https://doi.org/10.1061/JHEND8.HYENG-14239'},
    @{label='Krauklis'; uri='https://api.openalex.org/works?search=Hydraulic%20fracture%20diagnostics%20from%20Krauklis-wave%20resonance%20and%20tube-wave%20reflections&per-page=3'},
    @{label='recent-water-hammer'; uri='https://api.openalex.org/works?filter=title.search:water%20hammer,from_publication_date:2024-01-01,to_publication_date:2026-09-05&per-page=100'},
    @{label='water-hammer-operator'; uri='https://api.openalex.org/works?search=water%20hammer%20neural%20operator&per-page=15'},
    @{label='water-hammer-fracture-ML'; uri='https://api.openalex.org/works?search=water%20hammer%20hydraulic%20fracture%20machine%20learning&per-page=15'}
)
$records = foreach ($request in $requests) {
    try {
        # Hashtable parsing preserves case-distinct words in OpenAlex's abstract index.
        $response = (Invoke-WebRequest -Uri $request.uri -TimeoutSec 25).Content | ConvertFrom-Json -AsHashtable
        $works = if ($response.ContainsKey('results')) { $response.results } else { @($response) }
        $items = foreach ($work in $works) {
            $words = [System.Collections.Generic.SortedDictionary[int,string]]::new()
            if ($work.abstract_inverted_index) {
                foreach ($entry in $work.abstract_inverted_index.GetEnumerator()) {
                    foreach ($position in $entry.Value) { $words[[int]$position] = $entry.Key }
                }
            }
            @{
                title=$work.title; doi=$work.doi; date=$work.publication_date
                type=$work.type; authors=@($work.authorships | ForEach-Object { $_.author.display_name })
                venue=$work.primary_location.source.display_name
                abstract=($words.Values -join ' '); locations=$work.locations
            }
        }
        @{query=$request.label; uri=$request.uri; count=$response.meta.count; items=@($items)}
    } catch { @{query=$request.label; uri=$request.uri; error=$_.Exception.Message} }
}
@{retrieval_date='2026-09-05'; source='OpenAlex public API'; records=@($records)} |
    ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $OutputPath -Encoding utf8
foreach ($record in $records) {
    $selected = if ($record.query -eq 'recent-water-hammer') {
        @($record.items | Where-Object { $_.title -match 'neural|physics|learning|fractur|inverse|inversion' })
    } else { $record.items }
    @{query=$record.query; error=$record.error; items=@($selected | ForEach-Object {
        @{title=$_.title; doi=$_.doi; date=$_.date; abstract=$_.abstract}
    })} | ConvertTo-Json -Depth 6 -Compress
}
