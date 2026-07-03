import time
import numpy as np

from pymodaq.control_modules.viewer_utility_classes import comon_parameters, main
from pymodaq_data.data import DataToExport, DataWithAxes, Axis

from pymodaq_plugins_utils.hardware.camera_base_pylablib import CameraBasePyLabLib, cam_params
from pymodaq_plugins_thorlabs.utils import BeamAnalysisCache, analyze_beam_moments

try:
    from pylablib.devices import uc480
except Exception:
    uc480 = None


def list_uc480_serials():
    if uc480 is None:
        return []
    try:
        return [cam_info.serial_number for cam_info in uc480.list_cameras()]
    except Exception:
        return []


serial_numbers = list_uc480_serials()

serial_params = [
    {'title': 'Serial number:', 'name': 'serial_number', 'type': 'list',
     'limits': serial_numbers, 'value': serial_numbers[0] if serial_numbers else ''}
]


beam_analysis_params = [
    {'title': 'Beam Analysis:', 'name': 'beam_analysis', 'type': 'group', 'children': [
        {'title': 'Enable:', 'name': 'enable', 'type': 'bool', 'value': True},
        {'title': 'Threshold mode:', 'name': 'threshold_mode', 'type': 'list',
         'limits': ['Auto (Percentage)', 'Fixed (Absolute)'], 'value': 'Auto (Percentage)'},
        {'title': 'Threshold (%):', 'name': 'threshold_percent', 'type': 'float',
         'value': 10.0, 'limits': [0.0, 100.0]},
        {'title': 'Threshold value:', 'name': 'threshold_value', 'type': 'float',
         'value': 10.0},
        {'title': 'Analyse every N frames:', 'name': 'analyse_every', 'type': 'int',
         'value': 1, 'limits': [1, 1000]},
        {'title': 'UI update period (s):', 'name': 'ui_update_period', 'type': 'float',
         'value': 0.2, 'limits': [0.01, 10.0]},
        {'title': 'Beam detected:', 'name': 'beam_detected', 'type': 'led',
         'value': False, 'readonly': True},
    ]},
    {'title': 'Calculated Beam Properties:', 'name': 'calculated_properties', 'type': 'group', 'children': [
        {'title': 'Centroid X:', 'name': 'centroid_x', 'type': 'float',
         'value': 0.0, 'readonly': True},
        {'title': 'Centroid Y:', 'name': 'centroid_y', 'type': 'float',
         'value': 0.0, 'readonly': True},
        {'title': 'Waist U:', 'name': 'waist_u', 'type': 'float',
         'value': 0.0, 'readonly': True},
        {'title': 'Waist V:', 'name': 'waist_v', 'type': 'float',
         'value': 0.0, 'readonly': True},
        {'title': 'Orientation (deg):', 'name': 'theta_deg', 'type': 'float',
         'value': 0.0, 'readonly': True},
        {'title': 'Relative Power:', 'name': 'power', 'type': 'float',
         'value': 0.0, 'readonly': True},
    ]}
]


class DAQ_2DViewer_UC480(CameraBasePyLabLib):
    """
    Plugin UC480 / Thorlabs avec analyse de faisceau intégrée.
    """

    params = comon_parameters + serial_params + cam_params + beam_analysis_params

    def ini_attributes(self):
        super().ini_attributes()
        self.controller = None

        self._beam_cache = BeamAnalysisCache()
        self._frame_counter = 0
        self._last_ui_update = 0.0

        self._last_beam_result = {
            'detected': False,
            'centroid_x': 0.0,
            'centroid_y': 0.0,
            'waist_u': 0.0,
            'waist_v': 0.0,
            'theta_deg': 0.0,
            'power': 0.0,
        }

    def ini_detector_custom(self, controller=None):
        if uc480 is None:
            raise ImportError(
                "pylablib.devices.uc480 indisponible. "
                "Vérifie l'installation des drivers Thorlabs/IDS et de pylablib."
            )

        serial = self.settings.child('serial_number').value()

        if serial == '':
            raise Exception('No compatible Thorlabs UC480 camera was found.')

        if self.is_master:
            dev_id = uc480.UC480Camera.find_by_serial(serial)
            self.controller = uc480.UC480Camera(dev_id=dev_id)
        else:
            self.controller = controller

    def grab_data(self, Naverage=1, **kwargs):
        """
        Acquisition d'une image puis analyse de faisceau.
        """

        # À vérifier selon la version pylablib utilisée.
        # Pour les caméras pylablib, snap() est généralement disponible.
        img = self.controller.snap()

        img = np.asarray(img)

        # Certains drivers peuvent retourner des images couleur.
        # Ici on force en 2D pour l'analyse.
        if img.ndim == 3:
            # Simple conversion RGB -> mono.
            img = img.mean(axis=2)

        H, W = img.shape

        self._frame_counter += 1

        analyse_enabled = self.settings['beam_analysis', 'enable']
        analyse_every = self.settings['beam_analysis', 'analyse_every']

        if analyse_enabled and self._frame_counter % analyse_every == 0:
            result = analyze_beam_moments(
                img,
                cache=self._beam_cache,
                threshold_mode=self.settings['beam_analysis', 'threshold_mode'],
                threshold_percent=self.settings['beam_analysis', 'threshold_percent'],
                threshold_value=self.settings['beam_analysis', 'threshold_value'],
            )
            self._last_beam_result = result
        else:
            result = self._last_beam_result

        self._update_beam_ui(result)

        x_axis = Axis(
            label='x',
            units='pix',
            data=np.arange(W),
            index=1,
        )

        y_axis = Axis(
            label='y',
            units='pix',
            data=np.arange(H),
            index=0,
        )

        data_image = DataWithAxes(
            name='beam_image',
            source='raw',
            dim='Data2D',
            data=[img],
            axes=[y_axis, x_axis],
        )

        data_cx = DataWithAxes(
            name='centroid_x',
            source='calculated',
            dim='Data0D',
            data=[np.array([result['centroid_x']])]
        )

        data_cy = DataWithAxes(
            name='centroid_y',
            source='calculated',
            dim='Data0D',
            data=[np.array([result['centroid_y']])]
        )

        data_wu = DataWithAxes(
            name='waist_u',
            source='calculated',
            dim='Data0D',
            data=[np.array([result['waist_u']])]
        )

        data_wv = DataWithAxes(
            name='waist_v',
            source='calculated',
            dim='Data0D',
            data=[np.array([result['waist_v']])]
        )

        data_theta = DataWithAxes(
            name='theta_deg',
            source='calculated',
            dim='Data0D',
            data=[np.array([result['theta_deg']])]
        )

        data_power = DataWithAxes(
            name='power',
            source='calculated',
            dim='Data0D',
            data=[np.array([result['power']])]
        )

        dte = DataToExport(
            name='UC480_BeamAnalysis',
            data=[
                data_image,
                data_cx,
                data_cy,
                data_wu,
                data_wv,
                data_theta,
                data_power,
            ]
        )

        self.dte_signal.emit(dte)

    def _update_beam_ui(self, result):
        """
        Met à jour l'interface du plugin, mais pas à chaque frame.
        Important pour la stabilité et les performances.
        """

        now = time.perf_counter()
        period = self.settings['beam_analysis', 'ui_update_period']

        if now - self._last_ui_update < period:
            return

        self._last_ui_update = now

        self.settings.child('beam_analysis', 'beam_detected').setValue(
            bool(result['detected'])
        )

        self.settings.child('calculated_properties', 'centroid_x').setValue(
            float(result['centroid_x'])
        )
        self.settings.child('calculated_properties', 'centroid_y').setValue(
            float(result['centroid_y'])
        )
        self.settings.child('calculated_properties', 'waist_u').setValue(
            float(result['waist_u'])
        )
        self.settings.child('calculated_properties', 'waist_v').setValue(
            float(result['waist_v'])
        )
        self.settings.child('calculated_properties', 'theta_deg').setValue(
            float(result['theta_deg'])
        )
        self.settings.child('calculated_properties', 'power').setValue(
            float(result['power'])
        )

    def close(self):
        try:
            if self.controller is not None:
                self.controller.close()
        except Exception:
            pass


if __name__ == '__main__':
    main(__file__, init=False)