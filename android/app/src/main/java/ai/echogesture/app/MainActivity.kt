package ai.echogesture.app

import android.graphics.Color
import android.os.Bundle
import android.view.WindowManager
import androidx.activity.ComponentActivity
import androidx.activity.SystemBarStyle
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.viewModels
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.lifecycleScope
import ai.echogesture.app.engine.Bench
import ai.echogesture.app.engine.SignEngine
import ai.echogesture.app.ui.EchoApp
import ai.echogesture.app.ui.theme.EchoTheme
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch

class MainActivity : ComponentActivity() {
    private val vm: MainViewModel by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // The app is always dark, so system bar icons are always light.
        enableEdgeToEdge(SystemBarStyle.dark(Color.TRANSPARENT), SystemBarStyle.dark(Color.TRANSPARENT))
        val mode = intent.getStringExtra("bench")
        if (mode == null) {
            setContent { EchoTheme { EchoApp(vm) } }
            return
        }
        // Benchmark mode (engine/Bench.kt): started from adb, no camera, screen kept on.
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        val bench = Bench(this, vm.engine)
        // --ez ui true: the real screens over the replayed video, so captions can be looked at and tapped; the
        // camera stays off and only the replay feeds the engine.
        val ui = mode == "live" && intent.getBooleanExtra("ui", false)
        vm.engine.replayOnly = ui
        setContent {
            EchoTheme {
                if (ui) EchoApp(vm) else {
                    val status by bench.status.collectAsStateWithLifecycle()
                    Column(Modifier.fillMaxSize().padding(32.dp), verticalArrangement = Arrangement.Center) {
                        Text("EchoGesture benchmark", style = MaterialTheme.typography.headlineSmall)
                        Text(status, style = MaterialTheme.typography.bodyLarge, modifier = Modifier.padding(top = 12.dp))
                    }
                }
            }
        }
        lifecycleScope.launch {
            vm.settings.first { true }
            vm.engine.load()
            runCatching {
                when (mode) {
                    "model" -> bench.model(intent.getIntExtra("runs", 400))
                    "power" -> {
                        delay(intent.getIntExtra("wait", 45) * 1000L)   // time to unplug the phone
                        bench.live(150, 30, listOf(false, false), prefix = "power")
                        bench.live(90, 30, listOf(false, false), blank = true, gate = true, prefix = "power")
                        bench.live(90, 30, listOf(false, false), blank = true, gate = false, prefix = "power")
                        bench.live(90, 30, listOf(true), prefix = "power")
                    }
                    "powerall" -> {
                        // Everything the power rows need in one unplugged session, in the app's shipped settings:
                        // screen-on baseline, the classifier alone at the live rate (1 thread, not spinning), the live
                        // pipeline signing on 2 CPU lanes, and nobody in view at the signing-gated rate.
                        delay(intent.getIntExtra("wait", 60) * 1000L)   // time to unplug the phone
                        bench.powerSplit(intent.getIntExtra("seconds", 90), 1, false)
                        bench.live(150, 30, listOf(false, false), prefix = "power")
                        bench.live(90, 30, listOf(false, false), blank = true, gate = true, prefix = "power")
                    }
                    "powerpair" -> {
                        delay(intent.getIntExtra("wait", 60) * 1000L)   // time to unplug the phone
                        bench.powerPaired(intent.getIntExtra("rounds", 4), intent.getIntExtra("seconds", 40))
                        bench.live(150, 30, listOf(false, false), prefix = "power")
                    }
                    "power2" -> {
                        delay(intent.getIntExtra("wait", 45) * 1000L)
                        bench.powerSplit(intent.getIntExtra("seconds", 90), intent.getIntExtra("threads", 2), intent.getBooleanExtra("spin", false))
                    }
                    "clip" -> bench.clip(intent.getIntExtra("runs", 10), intent.getBooleanExtra("gpu", false), intent.getStringExtra("file") ?: "bench.mp4")
                    "teach" -> {
                        if (intent.hasExtra("continuous")) vm.engine.continuousForBench(intent.getBooleanExtra("continuous", true))
                        bench.teach((intent.getStringExtra("words") ?: "").split(",").map { it.trim() }.filter { it.isNotEmpty() },
                            intent.getIntExtra("examples", 3), intent.getStringExtra("file") ?: "teach_test.mp4",
                            intent.getIntExtra("fps", 25), intent.getIntExtra("seconds", 40), intent.getBooleanExtra("keep", false))
                    }
                    "live" -> {
                        val gpu = intent.getBooleanExtra("gpu", false)
                        val lanes = intent.getStringExtra("lanes")?.split(",")?.map { it.trim() == "gpu" }
                            ?: List(SignEngine.DEFAULT_LANES) { gpu }
                        vm.engine.practiceTarget = intent.getStringExtra("practice")   // Learn mode's check on a replayed video
                        if (intent.hasExtra("boundary")) vm.engine.boundaryForBench(intent.getBooleanExtra("boundary", true))
                        if (intent.hasExtra("continuous")) vm.engine.continuousForBench(intent.getBooleanExtra("continuous", true))
                        bench.live(intent.getIntExtra("seconds", 600), intent.getIntExtra("fps", 30), lanes,
                            blank = intent.getBooleanExtra("blank", false), gate = intent.getBooleanExtra("gate", false),
                            file = intent.getStringExtra("file") ?: "bench.mp4", voice = intent.getBooleanExtra("voice", false))
                    }
                }
            }.onFailure { android.util.Log.e(Bench.TAG, "benchmark failed", it) }
        }
    }
}
