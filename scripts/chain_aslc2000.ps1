# When every WLASL-vocabulary ASL Citizen clip is landmarked, build its cache and train the 2000-word models with it:
# a measurement run (ASL Citizen train/val signers only, so its test signers stay unseen) and two deployment runs.
$ErrorActionPreference = "Continue"
$py = "C:\Users\AASWAN~1\AppData\Local\Temp\xr2\m\Scripts\python.exe"
$log = "runs\log_chain_aslc2000.txt"
while ($true) {
    $n = (& $py -I scripts\aslc_cache.py data\aslcitizen\manifest.json data\kp_aslc data\cache_aslc2000.pt --check).Split(" ")
    "check $(Get-Date -Format HH:mm) $($n -join '/')" | Out-File $log -Append
    if ([int]$n[0] -ge [int]$n[1] - 20) { break }
    Start-Sleep -Seconds 120
}
& $py -I scripts\aslc_cache.py data\aslcitizen\manifest.json data\kp_aslc data\cache_aslc2000.pt --max-cls 2000 --t-max 48 *>> $log

$common = @("-I", "scripts\train_v2.py", "data\index\WLASL_v0.3.json", "data\cache_hol2000w.pt")
$opts = @("--subset", "2000", "--epochs", "30", "--batch", "128", "--lr", "5e-4", "--warmup", "2", "--canon", "--eval-every", "10",
          "--phonology", "data\asllex\signdata.csv", "--trainval", "--aug-cache", "data\cache_aug2000.pt", "--sparse", "0.3", "--t-max", "48")
function Launch($name, $seed, $init, $manifest) {
    Start-Process -FilePath $py -WindowStyle Hidden -RedirectStandardOutput "runs\log_$name.txt" -RedirectStandardError "runs\err_$name.txt" `
        -ArgumentList ($common + @("runs\$name") + $opts + @("--seed", "$seed", "--init", $init, "--extra", "data\cache_aslc2000.pt::$manifest"))
}
Launch "w2000_aslc_measure_s50" 50 "runs\w2000_tv_aug_ft_s23\best.pt" "data\aslcitizen\manifest_trainval.json"
Start-Sleep -Seconds 120
Launch "w2000_aslc_deploy_s51" 51 "runs\w2000_tv_aug_ft_s23\best.pt" "data\aslcitizen\manifest.json"
"launched measure s50 and deploy s51 $(Get-Date -Format HH:mm)" | Out-File $log -Append
while (-not (Test-Path runs\w2000_aslc_measure_s50\result.json)) { Start-Sleep -Seconds 60 }
Launch "w2000_aslc_deploy_s52" 52 "runs\w2000_tv_aug_ft_s24\best.pt" "data\aslcitizen\manifest.json"
"launched deploy s52 $(Get-Date -Format HH:mm)" | Out-File $log -Append
