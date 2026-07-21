"""
daq_2Dviewer_UC480.py
=====================
Plugin PyMoDAQ 5.1.x pour caméras Thorlabs / IDS basées sur uc480_64.dll.

Dépendance hardware :
    uc480_driver.py  (doit être dans le même dossier hardware/ ou dans le PYTHONPATH)

Structure du plugin dans le package PyMoDAQ :
    pymodaq_plugins_uc480/
    ├── hardware/
    │   └── uc480_driver.py
    └── daq_viewer_plugins/
        └── plugins_2D/
            └── daq_2Dviewer_UC480.py   ← ce fichier

Fonctionnalités :
    - Snap (single-shot) via FreezeVideo bloquant
    - Live (grab en boucle PyMoDAQ)
    - Averaging logiciel (Naverage)
    - Réglage exposition, framerate, pixel clock
    - AOI (x, y, w, h) avec réallocation buffer automatique
    - Axes physiques en µm (pixel_size du capteur)
    - Remise à jour des paramètres si le driver sature une valeur
"""

import numpy as np
from qtpy.QtCore import QThread

from pymodaq.control_modules.viewer_utility_classes import (
    DAQ_Viewer_base, comon_parameters, main
)
from pymodaq.control_modules.thread_commands import ThreadStatus
from pymodaq.utils.data import DataFromPlugins, Axis, DataToExport
from pymodaq_utils.utils import ThreadCommand

# Import du driver hardware
try:
    # 1. Si le package est installé en mode package
    from pymodaq_plugins_thorlabs.hardware.uc480_driver import UC480Camera
except ImportError:
    # 2. Si exécuté localement depuis le dossier pymodaq_plugins_thorlabs
    from hardware.uc480_driver import UC480Camera


# ─────────────────────────────────────────────────────────────────────────────
# Plugin
# ─────────────────────────────────────────────────────────────────────────────

class DAQ_2DViewer_Thorlabs_UC480(DAQ_Viewer_base):
    """
    Plugin PyMoDAQ 5 pour caméra UC480 (Thorlabs / IDS).

    Attributs PyMoDAQ
    -----------------
    live_mode_available : False
        Le live est géré par la boucle PyMoDAQ (snap répété).
        FreezeVideo bloque déjà pendant l'exposition — pas besoin de thread
        interne supplémentaire.
    hardware_averaging : False
        L'averaging est réalisé logiciellement ici (somme de N snaps).
    """

    live_mode_available  = False
    hardware_averaging   = False

    params = comon_parameters + [
        {
            'title': 'Caméra UC480', 'name': 'cam_group', 'type': 'group',
            'children': [
                {
                    'title': 'Index caméra', 'name': 'cam_index',
                    'type': 'int', 'value': 0, 'min': 0,
                    'tip': '0 = première caméra détectée automatiquement',
                },
                {
                    'title': 'Capteur', 'name': 'sensor_name',
                    'type': 'str', 'value': '', 'readonly': True,
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
                    'type': 'float', 'value': 10.0, 'min': 0.0,
                    'tip': 'Exposition en millisecondes. '
                           'Bornée à exp_max = 1/fps_min.',
                },
                {
                    'title': 'Expo min (ms)', 'name': 'exp_min_ms',
                    'type': 'float', 'value': 0.0, 'readonly': True,
                },
                {
                    'title': 'Expo max (ms)', 'name': 'exp_max_ms',
                    'type': 'float', 'value': 0.0, 'readonly': True,
                },
                {
                    'title': 'Framerate (fps)', 'name': 'framerate',
                    'type': 'float', 'value': 10.0, 'min': 0.0,
                    'tip': 'Framerate cible. Détermine la limite haute '
                           'd\'exposition.',
                },
                {
                    'title': 'FPS min', 'name': 'fps_min',
                    'type': 'float', 'value': 0.0, 'readonly': True,
                },
                {
                    'title': 'FPS max', 'name': 'fps_max',
                    'type': 'float', 'value': 0.0, 'readonly': True,
                },
            ],
        },
        {
            'title': 'Matériel', 'name': 'hw_group', 'type': 'group',
            'children': [
                {
                    'title': 'Pixel clock (MHz)', 'name': 'pixelclock_mhz',
                    'type': 'int', 'value': 24, 'min': 1,
                    'tip': 'Augmenter le pixel clock → FPS max plus élevé, '
                           'mais plus de bruit de lecture.',
                },
                {
                    'title': 'PC min (MHz)', 'name': 'pc_min',
                    'type': 'int', 'value': 0, 'readonly': True,
                },
                {
                    'title': 'PC max (MHz)', 'name': 'pc_max',
                    'type': 'int', 'value': 0, 'readonly': True,
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

    # ── PyMoDAQ lifecycle ─────────────────────────────────────────────────────

    def ini_attributes(self):
        """Initialisation des attributs internes (appelé par __init__ de la base)."""
        self.controller: UC480Camera = None
        self.x_axis = None
        self.y_axis = None
        self._pixel_size_um: float = 1.0   # mis à jour dans ini_detector

    def ini_detector(self, controller=None):
        """
        Ouvre la caméra, configure les paramètres par défaut, envoie
        un premier frame vide pour initialiser le viewer.

        Retourne (info: str, initialized: bool).
        """
        # ini_detector_init gère la logique Master/Slave PyMoDAQ
        self.ini_detector_init(
            old_controller=controller,
            new_controller=UC480Camera()
        )
        cam: UC480Camera = self.controller

        try:
            info_hw = cam.open(
                cam_index=self.settings.child('cam_group', 'cam_index').value()
            )
        except Exception as e:
            self.emit_status(
                ThreadCommand(ThreadStatus.UPDATE_STATUS,
                              f'UC480 open() échoué : {e}')
            )
            return f'Erreur : {e}', False

        # ── Peupler les paramètres en lecture seule ──────────────────────────
        self._pixel_size_um = info_hw.pixel_size_um

        self.settings.child('cam_group', 'sensor_name').setValue(info_hw.name)
        self.settings.child('cam_group', 'resolution').setValue(
            f'{info_hw.max_width} × {info_hw.max_height}'
        )
        self.settings.child('cam_group', 'pixel_size_um').setValue(
            info_hw.pixel_size_um
        )

        self.settings.child('acq_group', 'exp_min_ms').setValue(info_hw.exp_min_ms)
        self.settings.child('acq_group', 'exp_max_ms').setValue(info_hw.exp_max_ms)
        self.settings.child('acq_group', 'fps_min').setValue(
            round(info_hw.fps_min, 4)
        )
        self.settings.child('acq_group', 'fps_max').setValue(
            round(info_hw.fps_max, 4)
        )

        self.settings.child('hw_group', 'pc_min').setValue(info_hw.pc_min_mhz)
        self.settings.child('hw_group', 'pc_max').setValue(info_hw.pc_max_mhz)
        self.settings.child('hw_group', 'pixelclock_mhz').setValue(
            info_hw.pc_current_mhz
        )

        # Borner le widget exposition à la plage réelle
        self.settings.child('acq_group', 'exposure_ms').setOpts(
            min=info_hw.exp_min_ms,
            max=info_hw.exp_max_ms,
        )
        self.settings.child('acq_group', 'framerate').setOpts(
            min=info_hw.fps_min,
            max=info_hw.fps_max,
        )

        # AOI par défaut = plein capteur
        self.settings.child('aoi_group', 'aoi_w').setValue(info_hw.max_width)
        self.settings.child('aoi_group', 'aoi_h').setValue(info_hw.max_height)
        self.settings.child('aoi_group', 'aoi_x').setOpts(max=info_hw.max_width)
        self.settings.child('aoi_group', 'aoi_y').setOpts(max=info_hw.max_height)
        self.settings.child('aoi_group', 'aoi_w').setOpts(max=info_hw.max_width)
        self.settings.child('aoi_group', 'aoi_h').setOpts(max=info_hw.max_height)

        # ── Appliquer les paramètres initiaux ────────────────────────────────
        actual_fps = cam.set_framerate(
            self.settings.child('acq_group', 'framerate').value()
        )
        actual_exp = cam.set_exposure(
            self.settings.child('acq_group', 'exposure_ms').value()
        )
        self.settings.child('acq_group', 'framerate').setValue(round(actual_fps, 4))
        self.settings.child('acq_group', 'exposure_ms').setValue(round(actual_exp, 4))

        # ── Axes physiques (pixels → µm) ─────────────────────────────────────
        self._update_axes(cam.width, cam.height)

        # ── Initialiser le viewer avec un frame vide ─────────────────────────
        self.dte_signal_temp.emit(
            self._build_dte(
                np.zeros((cam.height, cam.width), dtype=np.uint8)
            )
        )

        self.emit_status(
            ThreadCommand(ThreadStatus.UPDATE_STATUS,
                          f'UC480 initialisée : {info_hw.name}  '
                          f'{info_hw.max_width}×{info_hw.max_height}')
        )
        return f'UC480 {info_hw.name}', True

    def close(self):
        """Ferme proprement la caméra."""
        if self.controller is not None:
            self.controller.close()
            self.controller = None

    def grab_data(self, Naverage: int = 1, **kwargs):
        """
        Acquisition d'un frame (ou moyenne de Naverage frames).

        PyMoDAQ appelle grab_data() en boucle pour le mode live.
        FreezeVideo bloque pendant l'exposition → pas de sleep artificiel
        nécessaire.

        Parameters
        ----------
        Naverage : int
            Nombre de frames à moyenner logiciellement.
        """
        cam: UC480Camera = self.controller

        try:
            if Naverage <= 1:
                arr = cam.snap().astype(np.float32)
            else:
                acc = cam.snap().astype(np.float32)
                for _ in range(Naverage - 1):
                    acc += cam.snap().astype(np.float32)
                arr = acc / Naverage

            self.dte_signal.emit(self._build_dte(arr))

        except Exception as e:
            self.emit_status(
                ThreadCommand(ThreadStatus.UPDATE_STATUS,
                              f'UC480 grab_data() erreur : {e}')
            )

    def stop(self):
        """Arrêt demandé par PyMoDAQ (bouton Stop)."""
        # Rien à faire : on est en single-shot, pas de thread interne
        return ''

    def commit_settings(self, param):
        """
        Router les changements de paramètres vers le driver.
        Appelé automatiquement à chaque modification dans le panneau Settings.
        """
        cam: UC480Camera = self.controller
        if cam is None or not cam.is_open:
            return

        name = param.name()

        # ── Pixel clock ──────────────────────────────────────────────────────
        if name == 'pixelclock_mhz':
            try:
                cam.set_pixelclock(param.value())
                # Le pixel clock change les plages de FPS et d'exposition
                # → relire et mettre à jour les paramètres
                info = cam.sensor_info
                fps_min, fps_max = cam._get_fps_range()
                exp_min, exp_max, _ = cam._get_exposure_range_normal()
                self.settings.child('acq_group', 'fps_min').setValue(
                    round(fps_min, 4)
                )
                self.settings.child('acq_group', 'fps_max').setValue(
                    round(fps_max, 4)
                )
                self.settings.child('acq_group', 'exp_min_ms').setValue(exp_min)
                self.settings.child('acq_group', 'exp_max_ms').setValue(exp_max)
                self.emit_status(
                    ThreadCommand(ThreadStatus.UPDATE_STATUS,
                                  f'Pixel clock → {param.value()} MHz')
                )
            except Exception as e:
                self.emit_status(
                    ThreadCommand(ThreadStatus.UPDATE_STATUS, str(e))
                )

        # ── Framerate ────────────────────────────────────────────────────────
        elif name == 'framerate':
            actual = cam.set_framerate(param.value())
            # Remettre la valeur réelle dans le widget sans déclencher
            # commit_settings à nouveau (blockSignals n'existe pas dans
            # pyqtgraph Parameter → on compare pour éviter la récursion)
            if abs(actual - param.value()) > 0.001:
                self.settings.child('acq_group', 'framerate').setValue(
                    round(actual, 4)
                )
            # Re-sync l'exposition max accessible
            _, exp_max, _ = cam._get_exposure_range_normal()
            self.settings.child('acq_group', 'exp_max_ms').setValue(exp_max)

        # ── Exposition ───────────────────────────────────────────────────────
        elif name == 'exposure_ms':
            actual = cam.set_exposure(param.value())
            if abs(actual - param.value()) > 0.001:
                # Le driver a saturé → on corrige le widget
                self.settings.child('acq_group', 'exposure_ms').setValue(
                    round(actual, 4)
                )
                self.emit_status(
                    ThreadCommand(
                        ThreadStatus.UPDATE_STATUS,
                        f'Exposition saturée à {actual:.3f} ms '
                        f'(max = {self.settings.child("acq_group","exp_max_ms").value():.1f} ms)'
                    )
                )

        # ── AOI ──────────────────────────────────────────────────────────────
        elif name in ('aoi_enable', 'aoi_x', 'aoi_y', 'aoi_w', 'aoi_h'):
            self._apply_aoi(cam)

    # ── Helpers privés ───────────────────────────────────────────────────────

    def _apply_aoi(self, cam: UC480Camera):
        """Lit les paramètres AOI courants et les applique au driver."""
        if self.settings.child('aoi_group', 'aoi_enable').value():
            x = self.settings.child('aoi_group', 'aoi_x').value()
            y = self.settings.child('aoi_group', 'aoi_y').value()
            w = self.settings.child('aoi_group', 'aoi_w').value()
            h = self.settings.child('aoi_group', 'aoi_h').value()
        else:
            # AOI désactivée → plein capteur
            info = cam.sensor_info
            x, y, w, h = 0, 0, info.max_width, info.max_height

        try:
            ex, ey, ew, eh = cam.set_aoi(x, y, w, h)
            # Remettre les valeurs effectives dans les widgets
            self.settings.child('aoi_group', 'aoi_x').setValue(ex)
            self.settings.child('aoi_group', 'aoi_y').setValue(ey)
            self.settings.child('aoi_group', 'aoi_w').setValue(ew)
            self.settings.child('aoi_group', 'aoi_h').setValue(eh)
            self._update_axes(ew, eh)
            self.emit_status(
                ThreadCommand(ThreadStatus.UPDATE_STATUS,
                              f'AOI → {ew}×{eh} @ ({ex},{ey})')
            )
        except Exception as e:
            self.emit_status(
                ThreadCommand(ThreadStatus.UPDATE_STATUS,
                              f'AOI erreur : {e}')
            )

    def _update_axes(self, width: int, height: int):
        """(Re)calcule les axes physiques en µm selon la taille d'AOI courante."""
        px = self._pixel_size_um
        # index=1 → axe des colonnes (X), index=0 → axe des lignes (Y)
        self.x_axis = Axis(
            label='x', units='µm',
            data=np.arange(width) * px,
            index=1
        )
        self.y_axis = Axis(
            label='y', units='µm',
            data=np.arange(height) * px,
            index=0
        )

    def _build_dte(self, arr: np.ndarray) -> DataToExport:
        """Construit le DataToExport PyMoDAQ à partir d'un array numpy."""
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


# ─────────────────────────────────────────────────────────────────────────────
# Lancement en standalone (debug hors DAQ_Viewer)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    main(__file__)
