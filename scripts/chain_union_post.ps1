# After the union runs finish: score the measurement run on the unseen ASL Citizen signers (all 3,075 signs, and
# restricted to the 2,000 WLASL signs to compare with the full model), then export the deployment run as a single
# INT8 model (a pair would put the bundle over 20 MB with MediaPipe).
$ErrorActionPreference = "Continue"
$py = "C:\Users\AASWAN~1\AppData\Local\Temp\xr2\m\Scripts\python.exe"
$log = "runs\log_chain_union_post.txt"
while (-not ((Test-Path runs\union_measure_s70\result.json) -and (Test-Path runs\union_deploy_s71\result.json))) { Start-Sleep -Seconds 60 }
"training done $(Get-Date -Format HH:mm)" | Out-File $log -Append
foreach ($n in @(3075, 2000)) {
    & $py -I scripts\eval_aslc.py data\cache_aslc_all.pt data\aslcitizen\manifest_test.json runs\union_measure_s70\best.pt `
        --subset $n --out "runs\union_measure_s70\unseen_$n.json" *>> $log
}
& $py -I scripts\export_v2.py data\index\WLASL_v0.3.json data\cache_hol2000w.pt runs\deploy\union runs\union_deploy_s71\best.pt `
    --subset 3075 --labels data\aslcitizen\union_labels.json *>> $log
"done $(Get-Date -Format HH:mm)" | Out-File $log -Append
