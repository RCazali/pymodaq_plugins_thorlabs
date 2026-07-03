import time
import numpy as np
from pymodaq.control_modules.viewer_utility_classes import DAQ_Viewer_base, main
from pymodaq_data.data import DataToExport, DataWithAxes, Axis

class DAQ_2DViewer_beamMock(DAQ_Viewer_base):
    """
    Plugin de test simulant un faisceau laser qui dérive thermiquement
    et intégrant l'analyse par moments d'ordre 1 et 2 de l'utilisateur.
    """
    # On reprend tes paramètres d'analyse et on ajoute la simulation de dérive
    params = [
        {'title': 'Simulation Drift:', 'name': 'sim_settings', 'type': 'group', 'children': [
            {'title': 'Base X0 (pix):', 'name': 'x0', 'type': 'float', 'value': 256.0},
            {'title': 'Base Y0 (pix):', 'name': 'y0', 'type': 'float', 'value': 256.0},
            {'title': 'Drift Speed X (pix/s):', 'name': 'drift_x', 'type': 'float', 'value': 0.15},  # Dérive lente
            {'title': 'Drift Speed Y (pix/s):', 'name': 'drift_y', 'type': 'float', 'value': -0.08},
            {'title': 'Oscillation Amp (pix):', 'name': 'osc_amp', 'type': 'float', 'value': 50.0},   # Vibrations
            {'title': 'Noise level:', 'name': 'noise', 'type': 'float', 'value': 8.0},
            {'title': 'Base Waist U:', 'name': 'base_wu', 'type': 'float', 'value': 45.0},
            {'title': 'Base Waist V:', 'name': 'base_wv', 'type': 'float', 'value': 30.0},
            {'title': 'Orientation (deg):', 'name': 'base_angle', 'type': 'float', 'value': 22.5},
        ]},
        {'title': 'Analysis Settings:', 'name': 'analysis_settings', 'type': 'group', 'children': [
            {'title': 'Threshold mode:', 'name': 'threshold_mode', 'type': 'list', 
             'limits': ['Auto (Percentage)', 'Fixed (Absolute)'], 'value': 'Auto (Percentage)'},
            {'title': 'Threshold (%):', 'name': 'threshold_percent', 'type': 'float', 'value': 10.0, 'limits': [0.0, 100.0]},
            {'title': 'Threshold (Value):', 'name': 'threshold_value', 'type': 'float', 'value': 10.0},
            {'title': 'Beam detected:', 'name': 'beam_detected', 'type': 'led', 'value': False, 'readonly': True},
        ]},
        {'title': 'Calculated Beam Properties (0D):', 'name': 'calculated_properties', 'type': 'group', 'children': [
            {'title': 'Centroid X:', 'name': 'centroid_x', 'type': 'float', 'value': 0.0, 'readonly': True},
            {'title': 'Centroid Y:', 'name': 'centroid_y', 'type': 'float', 'value': 0.0, 'readonly': True},
            {'title': 'Waist U (rx):', 'name': 'waist_u', 'type': 'float', 'value': 0.0, 'readonly': True},
            {'title': 'Waist V (ry):', 'name': 'waist_v', 'type': 'float', 'value': 0.0, 'readonly': True},
            {'title': 'Orientation (deg):', 'name': 'theta_deg', 'type': 'float', 'value': 0.0, 'readonly': True},
            {'title': 'Average Power:', 'name': 'Power', 'type': 'float', 'value': 0.0, 'readonly': True},
        ]}
    ]

    def __init__(self, parent=None, params_state=None):
        super().__init__(parent, params_state)
        self.controller = None
        self.ini_time = None
        self._cached_shape = None
        self._x_grid = self._y_grid = self._x_grid_sq = self._y_grid_sq = self._xy_grid = None

    def ini_detector(self, controller=None):
        self.controller = "BeamAnalysisMock_Controller"
        self.ini_time = time.perf_counter()
        
        # Grilles fixes pour une caméra de 512x512
        self.x_axis = np.arange(512, dtype=np.float64)
        self.y_axis = np.arange(512, dtype=np.float64)
        
        initialized = True
        info = "Mock Beam Camera Initialized"
        return info, initialized

    def grab_data(self, Naverage=1, **kwargs):
        # 1. Calcul du temps écoulé pour simuler la dérive temporelle
        t = time.perf_counter() - self.ini_time

        # 2. Récupération des paramètres de simulation
        drift_x = self.settings['sim_settings', 'drift_x']
        drift_y = self.settings['sim_settings', 'drift_y']
        osc_amp = self.settings['sim_settings', 'osc_amp']
        
        # Coordonnées du centre qui dérive + oscille
        x0 = self.settings['sim_settings', 'x0'] + (drift_x * t) + osc_amp * np.sin(0.3 * t)
        y0 = self.settings['sim_settings', 'y0'] + (drift_y * t) + osc_amp * np.cos(0.2 * t)
        
        wu = self.settings['sim_settings', 'base_wu']
        wv = self.settings['sim_settings', 'base_wv']
        theta_rad = np.radians(self.settings['sim_settings', 'base_angle'])
        noise = self.settings['sim_settings', 'noise']

        # 3. Génération de l'image du faisceau incliné (rotation)
        X, Y = np.meshgrid(self.x_axis, self.y_axis)
        X_rot = (X - x0) * np.cos(theta_rad) + (Y - y0) * np.sin(theta_rad)
        Y_rot = -(X - x0) * np.sin(theta_rad) + (Y - y0) * np.cos(theta_rad)
        
        img = 1000.0 * np.exp(-((X_rot)**2 / (2 * wu**2) + (Y_rot)**2 / (2 * wv**2)))
        img += np.random.normal(0, noise, img.shape)

        # === DEBUT DE TON ANALYSE (strictement identique à ton extension) ===
        H, W = img.shape

        if self._cached_shape != (H, W):
            self._cached_shape = (H, W)
            self._x_grid, self._y_grid = np.meshgrid(self.x_axis, self.y_axis)
            self._x_grid_sq, self._y_grid_sq = self._x_grid ** 2, self._y_grid ** 2
            self._xy_grid = self._x_grid * self._y_grid

        # Traitement du seuil
        mode = self.settings['analysis_settings', 'threshold_mode']
        if mode == 'Auto (Percentage)':
            th = img.max() * (self.settings['analysis_settings', 'threshold_percent'] / 100.0)
        else:
            th = self.settings['analysis_settings', 'threshold_value']

        I_th = np.copy(img)
        I_th[I_th < th] = 0.0

        P = I_th.sum()
        if P <= 0:
            self.settings.child('analysis_settings', 'beam_detected').setValue(False)
            # Envoi d'une trame vide/par défaut en cas de perte de faisceau
            self.emit_empty_data()
            return

        # Calcul des moments
        x_bar = np.sum(I_th * self._x_grid) / P
        y_bar = np.sum(I_th * self._y_grid) / P

        mu_xx = max(0.0, np.sum(I_th * self._x_grid_sq) / P - x_bar**2)
        mu_yy = max(0.0, np.sum(I_th * self._y_grid_sq) / P - y_bar**2)
        mu_xy = np.sum(I_th * self._xy_grid) / P - x_bar * y_bar

        diff = mu_xx - mu_yy
        term = np.sqrt(diff**2 + 4.0 * mu_xy**2)

        w_u = 2.0 * np.sqrt((mu_xx + mu_yy + term) / 2.0)
        w_v = 2.0 * np.sqrt((mu_xx + mu_yy - term) / 2.0)
        theta = 0.5 * np.arctan2(2.0 * mu_xy, diff)
        
        power = np.mean(img)

        # Mise à jour des paramètres UI du plugin (lecture seule)
        self.settings.child('analysis_settings', 'beam_detected').setValue(True)
        self.settings.child('calculated_properties', 'centroid_x').setValue(float(x_bar))
        self.settings.child('calculated_properties', 'centroid_y').setValue(float(y_bar))
        self.settings.child('calculated_properties', 'waist_u').setValue(float(w_u))
        self.settings.child('calculated_properties', 'waist_v').setValue(float(w_v))
        self.settings.child('calculated_properties', 'theta_deg').setValue(float(np.degrees(theta)))
        self.settings.child('calculated_properties', 'Power').setValue(float(power))
        # === FIN DE TON ANALYSE ===

        # 4. Construction et envoi du DataToExport complet (Image 2D + Données calculées 0D)
        axis_x = Axis(label='x', units='pix', data=self.x_axis, index=1)
        axis_y = Axis(label='y', units='pix', data=self.y_axis, index=0)

        # Image 2D
        data_image = DataWithAxes(
            name='beam_image',
            source='raw',
            dim='Data2D',
            data=[img],
            axes=[axis_y, axis_x]
        )

        # Les moments 0D (chacun dans son DataWithAxes 0D pour être routable)
        data_cx = DataWithAxes(name='centroid_x', source='calculated', dim='Data0D', data=[np.array([x_bar])])
        data_cy = DataWithAxes(name='centroid_y', source='calculated', dim='Data0D', data=[np.array([y_bar])])
        data_wu = DataWithAxes(name='waist_u', source='calculated', dim='Data0D', data=[np.array([w_u])])
        data_wv = DataWithAxes(name='waist_v', source='calculated', dim='Data0D', data=[np.array([w_v])])
        data_p  = DataWithAxes(name='power', source='calculated', dim='Data0D', data=[np.array([power])])

        # Export groupé
        dte = DataToExport(
            name='BeamAnalysisMock',
            data=[data_image, data_cx, data_cy, data_wu, data_wv, data_p]
        )

        self.dte_signal.emit(dte)

    def emit_empty_data(self):
        """Envoi de zéros si le faisceau n'est pas détecté."""
        empty_img = np.zeros((512, 512))
        data_image = DataWithAxes(name='beam_image', source='raw', dim='Data2D', data=[empty_img])
        data_cx = DataWithAxes(name='centroid_x', source='calculated', dim='Data0D', data=[np.array([0.0])])
        data_cy = DataWithAxes(name='centroid_y', source='calculated', dim='Data0D', data=[np.array([0.0])])
        data_wu = DataWithAxes(name='waist_u', source='calculated', dim='Data0D', data=[np.array([0.0])])
        data_wv = DataWithAxes(name='waist_v', source='calculated', dim='Data0D', data=[np.array([0.0])])
        data_p  = DataWithAxes(name='power', source='calculated', dim='Data0D', data=[np.array([0.0])])
        
        self.dte_signal.emit(DataToExport(name='BeamAnalysisMock', data=[data_image, data_cx, data_cy, data_wu, data_wv, data_p]))

    def close(self):
        pass

if __name__ == '__main__':
    main(DAQ_2DViewer_beamMock)