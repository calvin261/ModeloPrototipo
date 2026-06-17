package com.lsec.translator.ui.navigation

import androidx.compose.runtime.Composable
import androidx.navigation.NavHostController
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import com.lsec.translator.ui.screens.camera.CameraScreen
import com.lsec.translator.ui.screens.camera.CalibrationScreen
import com.lsec.translator.ui.screens.capture.DataCaptureScreen
import com.lsec.translator.ui.screens.dictionary.DictionaryScreen
import com.lsec.translator.ui.screens.home.HomeScreen
import com.lsec.translator.ui.screens.settings.SettingsScreen
import com.lsec.translator.ui.screens.training.TrainingScreen

/**
 * Rutas de navegación de la aplicación.
 */
sealed class Screen(val route: String) {
    data object Home : Screen("home")
    data object Camera : Screen("camera")
    data object Calibration : Screen("calibration")
    data object Dictionary : Screen("dictionary")
    data object Settings : Screen("settings")
    data object DataCapture : Screen("data_capture")
    data object Training : Screen("training")
}

/**
 * Composable principal de navegación.
 */
@Composable
fun LSECNavHost(
    navController: NavHostController = rememberNavController(),
    startDestination: String = Screen.Home.route
) {
    NavHost(
        navController = navController,
        startDestination = startDestination
    ) {
        composable(Screen.Home.route) {
            HomeScreen(
                onNavigateToCamera = { navController.navigate(Screen.Camera.route) },
                onNavigateToDictionary = { navController.navigate(Screen.Dictionary.route) },
                onNavigateToSettings = { navController.navigate(Screen.Settings.route) },
                onNavigateToDataCapture = { navController.navigate(Screen.DataCapture.route) },
                onNavigateToTraining = { navController.navigate(Screen.Training.route) }
            )
        }
        
        composable(Screen.Camera.route) {
            CameraScreen(
                onNavigateBack = { navController.popBackStack() },
                onNavigateToCalibration = { navController.navigate(Screen.Calibration.route) },
            )
        }

        composable(Screen.Calibration.route) {
            CalibrationScreen(
                onNavigateBack = { navController.popBackStack() }
            )
        }
        
        composable(Screen.Dictionary.route) {
            DictionaryScreen(
                onNavigateBack = { navController.popBackStack() }
            )
        }
        
        composable(Screen.Settings.route) {
            SettingsScreen(
                onNavigateBack = { navController.popBackStack() }
            )
        }

        composable(Screen.DataCapture.route) {
            DataCaptureScreen(
                onNavigateBack = { navController.popBackStack() }
            )
        }

        composable(Screen.Training.route) {
            TrainingScreen(
                onNavigateBack = { navController.popBackStack() }
            )
        }
    }
}
