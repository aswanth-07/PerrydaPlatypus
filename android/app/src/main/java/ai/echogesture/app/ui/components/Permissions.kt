package ai.echogesture.app.ui.components

import android.app.Activity
import android.content.Context
import android.content.ContextWrapper
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.provider.Settings
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.platform.LocalContext
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.LifecycleResumeEffect

/**
 * A runtime permission and what the person did with it. [blocked] means Android will not show its dialog again
 * (denied twice, or "Don't ask again"): asking does nothing, so the screen offers the app's settings page instead.
 */
class PermissionState(
    val granted: Boolean,
    val denied: Boolean,
    val blocked: Boolean,
    private val ask: () -> Unit,
    private val settings: () -> Unit,
) {
    /** Shows Android's dialog, or the app's settings page when the dialog would not appear. */
    fun request() = if (blocked) settings() else ask()
    fun openSettings() = settings()
}

@Composable
fun rememberPermission(permission: String, onResult: (granted: Boolean, blocked: Boolean) -> Unit = { _, _ -> }): PermissionState {
    val context = LocalContext.current
    val activity = remember(context) { context.findActivity() }
    val prefs = remember(context) { context.getSharedPreferences("permissions", Context.MODE_PRIVATE) }
    fun isGranted() = ContextCompat.checkSelfPermission(context, permission) == PackageManager.PERMISSION_GRANTED
    fun isBlocked() = !isGranted() && prefs.getBoolean(permission, false) && activity != null &&
        !ActivityCompat.shouldShowRequestPermissionRationale(activity, permission)

    var granted by remember { mutableStateOf(isGranted()) }
    var denied by remember { mutableStateOf(false) }
    var blocked by remember { mutableStateOf(isBlocked()) }
    val launcher = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { ok ->
        granted = ok
        denied = !ok
        blocked = !ok && isBlocked()
        onResult(ok, blocked)
    }
    // The person may have changed it in Android's settings while away.
    LifecycleResumeEffect(permission) {
        granted = isGranted()
        if (granted) { denied = false; blocked = false } else blocked = isBlocked()
        onPauseOrDispose { }
    }
    return PermissionState(
        granted, denied, blocked,
        ask = { prefs.edit().putBoolean(permission, true).apply(); launcher.launch(permission) },
        settings = { context.openAppSettings() },
    )
}

/** This app's page in Android's settings, where permissions can be turned back on. */
fun Context.openAppSettings() {
    startActivity(
        Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS, Uri.fromParts("package", packageName, null))
            .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
    )
}

tailrec fun Context.findActivity(): Activity? = when (this) {
    is Activity -> this
    is ContextWrapper -> baseContext.findActivity()
    else -> null
}
