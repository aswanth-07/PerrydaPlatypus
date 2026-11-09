# Runs the 2000-word steps as soon as the Holistic extraction finishes, so neither the GPU nor the CPU idles.
$ErrorActionPreference = "Continue"
$py = "C:\Users\AASWAN~1\AppData\Local\Temp\xr2\m\Scripts\python.exe"
$models = "C:\Users\AASWAN~1\AppData\Local\Temp\xr2\models"
while (-not (Select-String -Path runs\log_extract_hol3.txt -Pattern "^done" -Quiet)) { Start-Sleep -Seconds 30 }
"extraction done $(Get-Date)" | Out-File runs\log_chain_2000.txt

& $py -I scripts\data_v2.py data\kp_hol data\cache_hol2000w.pt --format holistic --index data\index\WLASL_v0.3.json --top 2000 *>> runs\log_chain_2000.txt

$train = @("-I", "scripts\train_v2.py", "data\index\WLASL_v0.3.json", "data\cache_hol2000w.pt")
foreach ($sd in 7, 8) {
    $out = "runs\w2000_holw_canon_phon_s$sd"
    Start-Process -FilePath $py -WindowStyle Hidden -RedirectStandardOutput "runs\log_w2000_holw_canon_phon_s$sd.txt" -RedirectStandardError "runs\err_w2000_holw_canon_phon_s$sd.txt" `
        -ArgumentList ($train + @($out, "--subset", "2000", "--epochs", "200", "--batch", "128", "--lr", "1.5e-3", "--seed", "$sd", "--canon", "--eval-every", "10", "--phonology", "data\asllex\signdata.csv"))
    Start-Sleep -Seconds 90
}
"training started $(Get-Date)" | Out-File runs\log_chain_2000.txt -Append

& $py -I scripts\extract_degraded.py train data\index\WLASL_v0.3.json data\wlasl21k\WLASL\videos data\kp_aug2000 $models --top 2000 --copies 1 --main-videos data\wlasl_main\videos --workers 18 *> runs\log_extract_aug2000.txt
"degraded extraction done $(Get-Date)" | Out-File runs\log_chain_2000.txt -Append
