"""
daq_2Dviewer_Thorlabs_UC480.py
==============================
Plugin PyMoDAQ 5.1.x pour caméras Thorlabs / IDS (UC480 / uEye).
"""

import numpy as np
import ctypes
from ctypes import c_int, byref, Structure

from pymodaq.control_modules.viewer_utility_classes import (
    DAQ_Viewer_base, comon_parameters, main
)
from pymodaq.control_modules.thread_commands import ThreadStatus
from pymodaq.utils.data import DataFromPlugins, Axis, DataToExport
from pymodaq_utils.utils import ThreadCommand

# Import du driver hardware
try:
    from pymodaq_plugins_thorlabs.hardware.uc480_driver import UC480Camera
except ImportError:
    from hardware.uc480_driver import UC480Camera


def get_available_cameras(dll_name: str = "uc480_64.dll") -> dict[str, int]:
    """
    Retourne un dictionnaire {numéro_de_série_réel: camera_id}
    en interrogeant chaque caméra individuellement.
    """
    cams = {}
    try:
        lib = ctypes.WinDLL(dll_name)
        lib.is_GetNumberOfCameras.argtypes = [ctypes.POINTER(c_int)]
        lib.is_GetNumberOfCameras.restype = c_int

        num_cams = c_int(0)
        if lib.is_GetNumberOfCameras(byref(num_cams)) == 0 and num_cams.value > 0:
            for cam_id in range(1, num_cams.value + 1):
                h_cam = c_int(cam_id | 0x8000)
                if lib.is_InitCamera(byref(h_cam), None) == 0:
                    class BOARDINFO(Structure):
                        _fields_ = [
                            ("SerNo", ctypes.c_char * 12),
                            ("ID", ctypes.c_char * 20),
                            ("Version", ctypes.c_char * 10),
                            ("Date", ctypes.c_char * 12),
                            ("Select", ctypes.c_byte),
                            ("Type", ctypes.c_byte),
                            ("Reserved", ctypes.c_char * 8),
                        ]
                    binfo = BOARDINFO()
                    if lib.is_GetCameraInfo(h_cam, byref(binfo)) == 0:
                        ser_no = binfo.SerNo.decode('utf-8', errors='ignore').strip('\x00').strip()
                        if ser_no:
                            cams[ser_no] = cam_id
                        else:
                            cams[f"Cam_{cam_id}"] = cam_id
                    else:
                        cams[f"Cam_{cam_id}"] = cam_id
                    lib.is_ExitCamera(h_cam)
                else:
                    cams[f"Cam_{cam_id} (Occupée)"] = cam_id
    except Exception as e:
        print(f"Erreur lors de la détection des N° de série : {e}")

    return cams if cams else {"Cam_1": 1}


# Détection immédiate au chargement du module pour que PyMoDAQ connaisse les limites immédiatement
INITIAL_CAMS = get_available_cameras()
INITIAL_SERIALS = list(INITIAL_CAMS.keys()) if INITIAL_CAMS else ['Aucune caméra']


# ─────────────────────────────────────────────────────────────────────────────
# Plugin
# ─────────────────────────────────────────────────────────────────────────────

class DAQ_2DViewer_Thorlabs_UC480(DAQ_Viewer_base):
    live_mode_available = False
    hardware_averaging = False

    params = comon_parameters + [
        {
            'title': 'Caméra UC480', 'name': 'cam_group', 'type': 'group',
            'children': [
                {
                    'title': 'N° de Série', 'name': 'cam_serial',
                    'type': 'list', 'limits': INITIAL_SERIALS,
                    'value': INITIAL_SERIALS[0] if INITIAL_SERIALS else '',
                    'tip': 'Sélectionnez le numéro de série de la caméra.',
                },
                {
                    'title': 'ID Interne', 'name': 'cam_id_display',
                    'type': 'int', 
                    'value': INITIAL_CAMS.get(INITIAL_SERIALS[0], 1) if INITIAL_SERIALS else 1,
                    'readonly': True,
                },
                {
                    'title': 'Résolution (px)', 'name': 'resolution',
                    'type': 'str', 'value': '', 'readonly': True,
                },
                {
                    'title': 'Pixel size (µm)', 'name': 'pixel_size_um',
                    'type': 'float', 'value': 0.0, 'readonly': True,
                },
            ],
        },
        {
            'title': 'Acquisition', 'name': 'acq_group', 'type': 'group',
            'children': [
                {
                    'title': 'Exposition (ms)', 'name': 'exposure_ms',
                    'type': 'float', 'value': 10.0, 'min': 0.01, 'step': 1.0,
                    'tip': 'Exposition en millisecondes.',
                },
            ],
        },
        {
            'title': 'AOI', 'name': 'aoi_group', 'type': 'group',
            'children': [
                {
                    'title': 'Activer AOI', 'name': 'aoi_enable',
                    'type': 'bool', 'value': False,
                },
                {
                    'title': 'X (px)', 'name': 'aoi_x',
                    'type': 'int', 'value': 0, 'min': 0,
                },
                {
                    'title': 'Y (px)', 'name': 'aoi_y',
                    'type': 'int', 'value': 0, 'min': 0,
                },
                {
                    'title': 'Largeur (px)', 'name': 'aoi_w',
                    'type': 'int', 'value': 1280, 'min': 4,
                },
                {
                    'title': 'Hauteur (px)', 'name': 'aoi_h',
                    'type': 'int', 'value': 1024, 'min': 4,
                },
            ],
        },
    ]

    def ini_attributes(self):
        self.controller: UC480Camera = None
        self.x_axis = None
        self.y_axis = None
        self._pixel_size_um: float = 1.0
        self.cams_dict: dict[str, int] = INITIAL_CAMS

    def refresh_cam_list(self):
        """Re-scanne les caméras au besoin tout en conservant la sélection courante."""
        try:
            self.cams_dict = get_available_cameras()
            serials = list(self.cams_dict.keys())
            if serials:
                current_val = self.settings.child('cam_group', 'cam_serial').value()
                self.settings.child('cam_group', 'cam_serial').setLimits(serials)
                
                if current_val in serials:
                    self.settings.child('cam_group', 'cam_serial').setValue(current_val)
                    self.settings.child('cam_group', 'cam_id_display').setValue(self.cams_dict[current_val])
                else:
                    self.settings.child('cam_group', 'cam_serial').setValue(serials[0])
                    self.settings.child('cam_group', 'cam_id_display').setValue(self.cams_dict[serials[0]])
            else:
                self.settings.child('cam_group', 'cam_serial').setLimits(['Aucune caméra'])
        except Exception as e:
            print(f"Erreur lors du rafraîchissement des caméras : {e}")

    def ini_detector(self, controller=None):
        self.ini_detector_init(
            old_controller=controller,
            new_controller=UC480Camera()
        )
        cam: UC480Camera = self.controller

        # Re-scanner au cas où une caméra aurait été branchée/débranchée
        self.refresh_cam_list()

        selected_serial = self.settings.child('cam_group', 'cam_serial').value()
        selected_cam_id = self.cams_dict.get(selected_serial, 1)
        self.settings.child('cam_group', 'cam_id_display').setValue(selected_cam_id)

        try:
            cam.open(camera_id=selected_cam_id)
        except Exception as e:
            try:
                cam.close()
            except Exception:
                pass
            self.controller = None
            msg = f"UC480 (N° Série {selected_serial} / ID {selected_cam_id}) open() échoué : {e}"
            self.emit_status(ThreadCommand(ThreadStatus.UPDATE_STATUS, msg))
            return msg, False

        # Ajuster l'exposition
        exp_init = self.settings.child('acq_group', 'exposure_ms').value()
        try:
            cam.set_exposure(exp_init)
        except Exception as e:
            print(f"Avertissement : impossible de régler l'exposition initiale : {e}")

        # Mettre à jour l'IHM
        self.settings.child('cam_group', 'resolution').setValue(f'{cam._width} × {cam._height}')
        self.settings.child('cam_group', 'pixel_size_um').setValue(self._pixel_size_um)
        self.settings.child('aoi_group', 'aoi_w').setValue(cam._width)
        self.settings.child('aoi_group', 'aoi_h').setValue(cam._height)

        self._update_axes(x_start=0, y_start=0, width=cam._width, height=cam._height)

        self.dte_signal_temp.emit(
            self._build_dte(np.zeros((cam._height, cam._width), dtype=np.float32))
        )

        msg = f"UC480 initialisée — N° Série: {selected_serial} (ID {selected_cam_id})"
        self.emit_status(ThreadCommand(ThreadStatus.UPDATE_STATUS, msg))
        return msg, True

    def close(self):
        if self.controller is not None:
            try:
                self.controller.close()
            except Exception:
                pass
            self.controller = None

    def grab_data(self, Naverage: int = 1, **kwargs):
        cam: UC480Camera = self.controller
        if cam is None or cam._cam == 0:
            return

        try:
            if Naverage <= 1:
                arr = cam.capture_image(timeout_ms=1000)
            else:
                acc = cam.capture_image(timeout_ms=1000).astype(np.float32)
                for _ in range(Naverage - 1):
                    acc += cam.capture_image(timeout_ms=1000).astype(np.float32)
                arr = acc / Naverage

            self.dte_signal.emit(self._build_dte(arr))

        except Exception as e:
            self.emit_status(
                ThreadCommand(ThreadStatus.UPDATE_STATUS, f'UC480 grab_data() erreur : {e}')
            )

    def stop(self):
        return ''

    def commit_settings(self, param):
        cam: UC480Camera = self.controller
        name = param.name()

        # 1. Sélection du Numéro de Série
        if name == 'cam_serial':
            s_val = param.value()
            if s_val in self.cams_dict:
                new_id = self.cams_dict[s_val]
                self.settings.child('cam_group', 'cam_id_display').setValue(new_id)

        # 2. Exposition
        elif name == 'exposure_ms':
            if cam is not None and cam._cam != 0:
                try:
                    exp_val = param.value()
                    cam.set_exposure(exp_val)
                    self.emit_status(ThreadCommand(ThreadStatus.UPDATE_STATUS, f"Exposition ajustée à {exp_val} ms"))
                except Exception as e:
                    self.emit_status(ThreadCommand(ThreadStatus.UPDATE_STATUS, f"Erreur exposition : {e}"))

        # 3. AOI
        elif name in ('aoi_enable', 'aoi_x', 'aoi_y', 'aoi_w', 'aoi_h'):
            if cam is not None and cam._cam != 0:
                self._apply_aoi(cam)

    def _apply_aoi(self, cam: UC480Camera):
        if self.settings.child('aoi_group', 'aoi_enable').value():
            x = self.settings.child('aoi_group', 'aoi_x').value()
            y = self.settings.child('aoi_group', 'aoi_y').value()
            w = self.settings.child('aoi_group', 'aoi_w').value()
            h = self.settings.child('aoi_group', 'aoi_h').value()
        else:
            x, y, w, h = 0, 0, cam._width, cam._height

        try:
            cam.set_aoi(x, y, w, h)
            self._update_axes(x, y, cam._width, cam._height)
        except Exception as e:
            self.emit_status(ThreadCommand(ThreadStatus.UPDATE_STATUS, f'AOI erreur : {e}'))

    def _update_axes(self, x_start: int, y_start: int, width: int, height: int):
        px = self._pixel_size_um
        self.x_axis = Axis(label='x', units='µm', data=(np.arange(width) + x_start) * px, index=1)
        self.y_axis = Axis(label='y', units='µm', data=(np.arange(height) + y_start) * px, index=0)

    def _build_dte(self, arr: np.ndarray) -> DataToExport:
        return DataToExport(
            name='UC480',
            data=[
                DataFromPlugins(
                    name='Camera',
                    data=[arr.astype(np.float32)],
                    dim='Data2D',
                    axes=[self.y_axis, self.x_axis],
                )
            ]
        )


if __name__ == '__main__':
    main(__file__)