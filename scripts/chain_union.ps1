# Union vocabulary (WLASL 2000 + ASL Citizen's other signs = 3075 words): once every ASL Citizen clip is landmarked
# and the 2000-word ASL Citizen models are done, train a measurement run (ASL Citizen test signers held out) and a
# deployment run, both warm-started from the 2000-word ASL Citizen deployment model.
$ErrorActionPreference = "Continue"
$py = "C:\Users\AASWAN~1\AppData\Local\Temp\xr2\m\Scripts\python.exe"
$log = "runs\log_chain_union.txt"
while ($true) {
    $n = (& $py -I scripts\aslc_cache.py data\aslcitizen\manifest.json data\kp_aslc data\cache_aslc_all.pt --max-cls 99999 --check).Split(" ")
    "check $(Get-Date -Format HH:mm) $($n -join '/')" | Out-File $log -Append
    if ([int]$n[0] -ge [int]$n[1] - 30) { break }
    Start-Sleep -Seconds 180
}
while (-not ((Test-Path runs\w2000_aslc_deploy_s51\result.json) -and (Test-Path runs\w2000_aslc_deploy_s52\result.json))) { Start-Sleep -Seconds 60 }
& $py -I scripts\aslc_cache.py data\aslcitizen\manifest.json data\kp_aslc data\cache_aslc_all.pt --max-cls 99999 --t-max 48 *>> $log

$classes = 3075
$common = @("-I", "scripts\train_v2.py", "data\index\WLASL_v0.3.json", "data\cache_hol2000w.pt")
$opts = @("--subset", "$classes", "--epochs", "30", "--batch", "128", "--lr", "6e-4", "--warmup", "2", "--canon", "--eval-every", "10",
          "--phonology", "data\asllex\signdata.csv", "--trainval", "--aug-cache", "data\cache_aug2000.pt", "--sparse", "0.3", "--t-max", "48",
          "--init", "runs\w2000_aslc_deploy_s51\best.pt")
foreach ($run in @(@("union_measure_s70", 70, "data\aslcitizen\manifest_trainval.json"), @("union_deploy_s71", 71, "data\aslcitizen\manifest.json"))) {
    Start-Process -FilePath $py -WindowStyle Hidden -RedirectStandardOutput "runs\log_$($run[0]).txt" -RedirectStandardError "runs\err_$($run[0]).txt" `
        -ArgumentList ($common + @("runs\$($run[0])") + $opts + @("--seed", "$($run[1])", "--extra", "data\cache_aslc_all.pt::$($run[2])"))
    Start-Sleep -Seconds 150
}
"launched union runs $(Get-Date -Format HH:mm)" | Out-File $log -Append
