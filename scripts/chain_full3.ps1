# Full (2,000-word) models warm-started from the 3,075-word union model, the step that gave the core model +1.9,
# plus clean unseen-signer measurements. Measurement runs must descend only from runs that never saw ASL Citizen's
# test signers (manifest_trainval): union_measure_s70 started from a deployment model, so its unseen score is
# void; union_measure_clean_s72 starts from w2000_aslc_measure_s50 instead, and the full measurement from s72.
$ErrorActionPreference = "Continue"
$py = "C:\Users\AASWAN~1\AppData\Local\Temp\xr2\m\Scripts\python.exe"
$log = "runs\log_chain_full3.txt"
$common = @("-I", "scripts\train_v2.py", "data\index\WLASL_v0.3.json", "data\cache_hol2000w.pt")
$base = @("--epochs", "30", "--batch", "128", "--warmup", "2", "--canon", "--eval-every", "10", "--phonology", "data\asllex\signdata.csv",
          "--trainval", "--aug-cache", "data\cache_aug2000.pt", "--sparse", "0.3", "--t-max", "48")
function Launch($name, $seed, $init, $extra, $classes, $lr) {
    Start-Process -FilePath $py -WindowStyle Hidden -RedirectStandardOutput "runs\log_$name.txt" -RedirectStandardError "runs\err_$name.txt" `
        -ArgumentList ($common + @("runs\$name", "--subset", "$classes", "--lr", $lr) + $base + @("--seed", "$seed", "--init", $init, "--extra", $extra))
}
Launch "union_measure_clean_s72" 72 "runs\w2000_aslc_measure_s50\best.pt" "data\cache_aslc_all.pt::data\aslcitizen\manifest_trainval.json" 3075 "6e-4"
Start-Sleep -Seconds 150
Launch "w2000_fromunion_s90" 90 "runs\union_deploy_s71\best.pt" "data\cache_aslc2000.pt::data\aslcitizen\manifest.json" 2000 "5e-4"
Start-Sleep -Seconds 150
Launch "w2000_fromunion_s91" 91 "runs\union_deploy_s71\best.pt" "data\cache_aslc2000.pt::data\aslcitizen\manifest.json" 2000 "5e-4"
"launched s72 s90 s91 $(Get-Date -Format HH:mm)" | Out-File $log -Append
while (-not (Test-Path runs\union_measure_clean_s72\result.json)) { Start-Sleep -Seconds 60 }
& $py -I scripts\eval_aslc.py data\cache_aslc_all.pt data\aslcitizen\manifest_test.json runs\union_measure_clean_s72\best.pt `
    --subset 3075 --out runs\union_measure_clean_s72\unseen_3075.json *>> $log
& $py -I scripts\eval_aslc.py data\cache_aslc_all.pt data\aslcitizen\manifest_test.json runs\union_measure_clean_s72\best.pt `
    --subset 2000 --out runs\union_measure_clean_s72\unseen_2000.json *>> $log
Launch "w2000_fromunion_measure_s92" 92 "runs\union_measure_clean_s72\best.pt" "data\cache_aslc2000.pt::data\aslcitizen\manifest_trainval.json" 2000 "5e-4"
"launched measure s92 $(Get-Date -Format HH:mm)" | Out-File $log -Append
while (-not (Test-Path runs\w2000_fromunion_measure_s92\result.json)) { Start-Sleep -Seconds 60 }
& $py -I scripts\eval_aslc.py data\cache_aslc2000.pt data\aslcitizen\manifest_test.json runs\w2000_fromunion_measure_s92\best.pt `
    --subset 2000 --out runs\w2000_fromunion_measure_s92\unseen.json *>> $log
"done $(Get-Date -Format HH:mm)" | Out-File $log -Append
