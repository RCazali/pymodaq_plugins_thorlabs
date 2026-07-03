import time
import numpy as np
from pymodaq.control_modules.viewer_utility_classes import DAQ_Viewer_base, main
from pymodaq_data.data import DataToExport, DataWithAxes, Axis

# Import de ton module d'analyse optimisé
from pymodaq_plugins_thorlabs.utils import BeamAnalysisCache, analyze_beam_moments


class DAQ_2DViewer_beamMock(DAQ_Viewer_base):
    """
    Plugin de test simulant un faisceau laser qui dérive thermiquement.
    Utilise le module d'analyse optimisé pour calculer les moments.
    """
    
    params = [
        {'title': 'Simulation Drift:', 'name': 'sim_settings', 'type': 'group', 'children': [
            {'title': 'Base X0 (pix):', 'name': 'x0', 'type': 'float', 'value': 256.0},
            {'title': 'Base Y0 (pix):', 'name': 'y0', 'type': 'float', 'value': 256.0},
            {'title': 'Drift Speed X (pix/s):', 'name': 'drift_x', 'type': 'float', 'value': 0.15},
            {'title': 'Drift Speed Y (pix/s):', 'name': 'drift_y', 'type': 'float', 'value': -0.08},
            {'title': 'Oscillation Amp (pix):', 'name': 'osc_amp', 'type': 'float', 'value': 50.0},
            {'title': 'Noise level:', 'name': 'noise', 'type': 'float', 'value': 8.0},
            {'title': 'Base Waist U (sigma):', 'name': 'base_wu', 'type': 'float', 'value': 45.0},
            {'title': 'Base Waist V (sigma):', 'name': 'base_wv', 'type': 'float', 'value': 30.0},
            {'title': 'Orientation (deg):', 'name': 'base_angle', 'type': 'float', 'value': 22.5},
            # Paramètre pour stress-tester la résolution
            {'title': 'Image Size (px):', 'name': 'img_size', 'type': 'int', 'value': 512, 'limits': [64, 4096]},
        ]},
        {'title': 'Analysis Settings:', 'name': 'analysis_settings', 'type': 'group', 'children': [
            {'title': 'Threshold mode:', 'name': 'threshold_mode', 'type': 'list', 
             'limits': ['Auto (Percentage)', 'Fixed (Absolute)'], 'value': 'Auto (Percentage)'},
            {'title': 'Threshold (%):', 'name': 'threshold_percent', 'type': 'float', 'value': 10.0, 'limits': [0.0, 100.0]},
            {'title': 'Threshold (Value):', 'name': 'threshold_value', 'type': 'float', 'value': 10.0},
            {'title': 'Analyse every N frames:', 'name': 'analyse_every', 'type': 'int', 'value': 1, 'limits': [1, 1000]},
            {'title': 'UI update period (s):', 'name': 'ui_update_period', 'type': 'float', 'value': 0.1, 'limits': [0.01, 10.0]},
            {'title': 'Beam detected:', 'name': 'beam_detected', 'type': 'led', 'value': False, 'readonly': True},
        ]},
        {'title': 'Calculated Beam Properties (0D):', 'name': 'calculated_properties', 'type': 'group', 'children': [
            {'title': 'Centroid X:', 'name': 'centroid_x', 'type': 'float', 'value': 0.0, 'readonly': True},
            {'title': 'Centroid Y:', 'name': 'centroid_y', 'type': 'float', 'value': 0.0, 'readonly': True},
            {'title': 'Waist U (2*sigma):', 'name': 'waist_u', 'type': 'float', 'value': 0.0, 'readonly': True},
            {'title': 'Waist V (2*sigma):', 'name': 'waist_v', 'type': 'float', 'value': 0.0, 'readonly': True},
            {'title': 'Orientation (deg):', 'name': 'theta_deg', 'type': 'float', 'value': 0.0, 'readonly': True},
            {'title': 'Relative Power:', 'name': 'Power', 'type': 'float', 'value': 0.0, 'readonly': True},
        ]}
    ]

    def __init__(self, parent=None, params_state=None):
        super().__init__(parent, params_state)
        self.controller = None
        self.ini_time = None
        
        # Initialisation du cache pour l'analyse (évite les allocations mémoire)
        self._beam_cache = BeamAnalysisCache()
        self._frame_counter = 0
        self._last_ui_update = 0.0
        
        # Dictionnaire par défaut si le faisceau n'est pas détecté
        self._last_beam_result = {
            'detected': False, 'centroid_x': 0.0, 'centroid_y': 0.0,
            'waist_u': 0.0, 'waist_v': 0.0, 'theta_deg': 0.0, 'power': 0.0
        }

    def ini_detector(self, controller=None):
        self.controller = "BeamAnalysisMock_Controller"
        self.ini_time = time.perf_counter()
        initialized = True
        info = "Mock Beam Camera Initialized with Optimized Analysis"
        return info, initialized

    def grab_data(self, Naverage=1, **kwargs):
        # 1. Paramètres de simulation
        t = time.perf_counter() - self.ini_time
        size = self.settings['sim_settings', 'img_size']
        
        drift_x = self.settings['sim_settings', 'drift_x']
        drift_y = self.settings['sim_settings', 'drift_y']
        osc_amp = self.settings['sim_settings', 'osc_amp']
        
        x0 = self.settings['sim_settings', 'x0'] + (drift_x * t) + osc_amp * np.sin(0.3 * t)
        y0 = self.settings['sim_settings', 'y0'] + (drift_y * t) + osc_amp * np.cos(0.2 * t)
        
        wu = self.settings['sim_settings', 'base_wu']
        wv = self.settings['sim_settings', 'base_wv']
        theta_rad = np.radians(self.settings['sim_settings', 'base_angle'])
        noise = self.settings['sim_settings', 'noise']

        # 2. Génération de l'image (Simulation du hardware)
        # Note: On garde le meshgrid ICI car c'est la simulation de la caméra
        x_axis = np.arange(size, dtype=np.float64)
        y_axis = np.arange(size, dtype=np.float64)
        X, Y = np.meshgrid(x_axis, y_axis)
        
        X_rot = (X - x0) * np.cos(theta_rad) + (Y - y0) * np.sin(theta_rad)
        Y_rot = -(X - x0) * np.sin(theta_rad) + (Y - y0) * np.cos(theta_rad)
        
        img = 1000.0 * np.exp(-((X_rot)**2 / (2 * wu**2) + (Y_rot)**2 / (2 * wv**2)))
        img += np.random.normal(0, noise, img.shape)

        # 3. ANALYSE DU FAISCEAU (Optimisée)
        self._frame_counter += 1
        analyse_every = self.settings['analysis_settings', 'analyse_every']
        
        # On n'analyse pas forcément toutes les images pour économiser le CPU
        if self._frame_counter % analyse_every == 0:
            result = analyze_beam_moments(
                img,
                cache=self._beam_cache,
                threshold_mode=self.settings['analysis_settings', 'threshold_mode'],
                threshold_percent=self.settings['analysis_settings', 'threshold_percent'],
                threshold_value=self.settings['analysis_settings', 'threshold_value'],
            )
            self._last_beam_result = result
        else:
            result = self._last_beam_result

        # 4. MISE À JOUR DE L'UI (Throttlée pour ne pas freezer PyMoDAQ)
        now = time.perf_counter()
        ui_period = self.settings['analysis_settings', 'ui_update_period']
        
        if now - self._last_ui_update > ui_period:
            self._last_ui_update = now
            self.settings.child('analysis_settings', 'beam_detected').setValue(bool(result['detected']))
            self.settings.child('calculated_properties', 'centroid_x').setValue(float(result['centroid_x']))
            self.settings.child('calculated_properties', 'centroid_y').setValue(float(result['centroid_y']))
            self.settings.child('calculated_properties', 'waist_u').setValue(float(result['waist_u']))
            self.settings.child('calculated_properties', 'waist_v').setValue(float(result['waist_v']))
            self.settings.child('calculated_properties', 'theta_deg').setValue(float(result['theta_deg']))
            self.settings.child('calculated_properties', 'Power').setValue(float(result['power']))

        # 5. EXPORT DES DONNÉES
        axis_x = Axis(label='x', units='pix', data=x_axis, index=1)
        axis_y = Axis(label='y', units='pix', data=y_axis, index=0)

        data_image = DataWithAxes(name='beam_image', source='raw', dim='Data2D', data=[img], axes=[axis_y, axis_x])
        
        # Données 0D (toujours émises, même si l'analyse a été sautée, on envoie la dernière valeur connue)
        data_cx = DataWithAxes(name='centroid_x', source='calculated', dim='Data0D', data=[np.array([result['centroid_x']])])
        data_cy = DataWithAxes(name='centroid_y', source='calculated', dim='Data0D', data=[np.array([result['centroid_y']])])
        data_wu = DataWithAxes(name='waist_u', source='calculated', dim='Data0D', data=[np.array([result['waist_u']])])
        data_wv = DataWithAxes(name='waist_v', source='calculated', dim='Data0D', data=[np.array([result['waist_v']])])
        data_th = DataWithAxes(name='theta_deg', source='calculated', dim='Data0D', data=[np.array([result['theta_deg']])])
        data_p  = DataWithAxes(name='power', source='calculated', dim='Data0D', data=[np.array([result['power']])])

        dte = DataToExport(
            name='BeamAnalysisMock',
            data=[data_image, data_cx, data_cy, data_wu, data_wv, data_th, data_p]
        )

        self.dte_signal.emit(dte)

    def close(self):
        pass

if __name__ == '__main__':
    main(__file__, init=False)