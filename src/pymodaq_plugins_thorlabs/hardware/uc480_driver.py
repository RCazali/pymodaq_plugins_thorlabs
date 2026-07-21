"""
uc480_driver.py
===============
Classe bas-niveau pour caméras Thorlabs/IDS basées sur uc480_64.dll.
Conçue pour être utilisée par le plugin PyMoDAQ daq_2Dviewer_UC480.

Caractéristiques du C1285R12M (testé) :
  - Résolution max : 1280 × 1024
  - Exposition max en live : ~205 ms  (limité par fps_min ≈ 4.87 Hz)
  - Long exposure hardware : NON supporté (caps = 0x01)
  - Pixel clock : 5–43 MHz

Usage autonome (hors PyMoDAQ) :
    cam = UC480Camera()
    info = cam.open()
    cam.set_exposure(50.0)
    arr  = cam.snap()           # numpy uint8 (H, W)
    cam.close()
"""

import ctypes
import time
import numpy as np
from ctypes import (
    Structure, POINTER, byref, sizeof,
    c_int, c_int32, c_uint, c_uint16, c_uint32,
    c_double, c_char, c_char_p, c_void_p,
)
from dataclasses import dataclass, field
from typing import Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Structures ctypes miroir du header uc480.h
# ─────────────────────────────────────────────────────────────────────────────

class _SENSORINFO(Structure):
    _fields_ = [
        ("SensorID",             c_uint16),
        ("strSensorName",        c_char * 32),
        ("nColorMode",           c_char),
        ("nMaxWidth",            c_uint32),
        ("nMaxHeight",           c_uint32),
        ("bMasterGain",          c_int32),
        ("bRGain",               c_int32),
        ("bGGain",               c_int32),
        ("bBGain",               c_int32),
        ("bGlobShutter",         c_int32),
        ("wPixelSize",           c_uint16),
        ("nUpperLeftBayerPixel", c_char),
        ("Reserved",             c_char * 13),
    ]


class _IS_RECT(Structure):
    _fields_ = [
        ("s32X",      c_int32),
        ("s32Y",      c_int32),
        ("s32Width",  c_int32),
        ("s32Height", c_int32),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Dataclass de retour : informations capteur
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SensorInfo:
    sensor_id:      int
    name:           str
    max_width:      int
    max_height:     int
    pixel_size_um:  float          # µm
    global_shutter: bool
    long_exp_supported: bool = False
    exp_min_ms:     float = 0.0
    exp_max_ms:     float = 0.0    # limite hardware (= 1/fps_min - overhead)
    fps_min:        float = 0.0
    fps_max:        float = 0.0
    pc_min_mhz:     int   = 0
    pc_max_mhz:     int   = 0
    pc_current_mhz: int   = 0


# ─────────────────────────────────────────────────────────────────────────────
# Constantes uc480
# ─────────────────────────────────────────────────────────────────────────────

IS_SUCCESS            = 0
IS_CM_MONO8           = 6
IS_WAIT               = 1
IS_DONT_WAIT          = 0
IS_SET_TRIGGER_OFF    = 0x0000

# Exposure commands
_EXP_GET_CAPS         = 1
_EXP_GET_DEFAULT      = 2
_EXP_GET_RANGE_MIN    = 3
_EXP_GET_RANGE_MAX    = 4
_EXP_GET_RANGE_INC    = 5
_EXP_GET_CURRENT      = 7
_EXP_SET              = 12
_EXP_GET_LONG_MIN     = 13
_EXP_GET_LONG_MAX     = 14
_EXP_GET_LONG_INC     = 15
_EXP_GET_LONG_ENABLE  = 17
_EXP_SET_LONG_ENABLE  = 18
_EXP_CAP_LONG         = 0x00000004

# Pixel clock commands
_PC_GET_RANGE   = 3
_PC_GET_DEFAULT = 4
_PC_GET         = 5
_PC_SET         = 6

# AOI commands
IS_AOI_IMAGE_SET_AOI  = 0x0001
IS_AOI_IMAGE_GET_AOI  = 0x0002


# ─────────────────────────────────────────────────────────────────────────────
# Classe principale
# ─────────────────────────────────────────────────────────────────────────────

class UC480Camera:
    """
    Interface Python vers uc480_64.dll.

    État interne :
        _cam        : handle entier renvoyé par is_InitCamera
        _mem        : pointeur char* vers le buffer alloué dans le driver
        _mem_id     : identifiant du buffer
        _width/_height : dimensions de l'AOI courante
        _is_live    : True si CaptureVideo est actif
        _long_mode  : True si le mode longue exposition hardware est activé
        sensor_info : SensorInfo (peuplé par open())
    """

    DLL_NAME = "uc480_64.dll"

    def __init__(self):
        self._lib: Optional[ctypes.WinDLL] = None
        self._cam:    int       = 0
        self._mem:    c_char_p  = c_char_p()
        self._mem_id: int       = 0
        self._width:  int       = 0
        self._height: int       = 0
        self._is_live: bool     = False
        self._long_mode: bool   = False
        self.sensor_info: Optional[SensorInfo] = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def open(self, cam_index: int = 0) -> SensorInfo:
        """
        Ouvre la caméra, configure le mode mono8, alloue la mémoire,
        désactive le trigger. Retourne un SensorInfo complet.

        Parameters
        ----------
        cam_index : int
            0 = première caméra disponible (comportement du driver).
            Pour un index spécifique, passer l'ID caméra (1-based).
        """
        self._lib = ctypes.WinDLL(self.DLL_NAME)
        self._register_prototypes()

        # Init
        hCam = c_int(cam_index)          # 0 → auto-detect première caméra
        self._check(self._lib.is_InitCamera(byref(hCam), None), "InitCamera")
        self._cam = hCam.value

        # Sensor info
        si = _SENSORINFO()
        self._check(self._lib.is_GetSensorInfo(self._cam, byref(si)), "GetSensorInfo")
        self._width  = si.nMaxWidth
        self._height = si.nMaxHeight

        # Color mode
        self._check(self._lib.is_SetColorMode(self._cam, IS_CM_MONO8), "SetColorMode")

        # Trigger off (free-run)
        self._check(
            self._lib.is_SetExternalTrigger(self._cam, IS_SET_TRIGGER_OFF),
            "SetExternalTrigger OFF"
        )

        # Allocation mémoire
        self._alloc_mem(self._width, self._height)

        # Framerate / exposition ranges
        fps_min, fps_max = self._get_fps_range()
        exp_min, exp_max, _ = self._get_exposure_range_normal()
        caps = self._exposure_caps()
        long_supported = bool(caps & _EXP_CAP_LONG)

        # Pixel clock
        pc_min, pc_max, _ = self._get_pixelclock_range()
        pc_cur = self._get_pixelclock()

        self.sensor_info = SensorInfo(
            sensor_id       = si.SensorID,
            name            = si.strSensorName.decode(),
            max_width       = si.nMaxWidth,
            max_height      = si.nMaxHeight,
            pixel_size_um   = si.wPixelSize / 100.0,
            global_shutter  = bool(si.bGlobShutter),
            long_exp_supported = long_supported,
            exp_min_ms      = exp_min,
            exp_max_ms      = exp_max,
            fps_min         = fps_min,
            fps_max         = fps_max,
            pc_min_mhz      = pc_min,
            pc_max_mhz      = pc_max,
            pc_current_mhz  = pc_cur,
        )
        return self.sensor_info

    def close(self):
        """Arrête l'acquisition, libère la mémoire, ferme la caméra."""
        if self._lib is None:
            return
        if self._is_live:
            self.stop_live()
        try:
            self._lib.is_FreeImageMem(self._cam, self._mem, self._mem_id)
        except Exception:
            pass
        try:
            self._lib.is_ExitCamera(self._cam)
        except Exception:
            pass
        self._lib     = None
        self._cam     = 0
        self._is_live = False

    # ── Acquisition ───────────────────────────────────────────────────────────

    def snap(self) -> np.ndarray:
        """
        Capture un unique frame (FreezeVideo + IS_WAIT) et le retourne
        sous forme de numpy array uint8 de shape (height, width).

        C'est le mode privilégié pour PyMoDAQ : on laisse le driver
        bloquer le thread pendant la durée d'exposition, puis on copie.

        Note : si le live est actif, il est arrêté avant la capture.
        """
        if self._is_live:
            self.stop_live()

        ret = self._lib.is_FreezeVideo(self._cam, IS_WAIT)
        if ret != IS_SUCCESS:
            raise RuntimeError(f"is_FreezeVideo failed: error {ret}")

        return self._copy_to_numpy()

    def start_live(self):
        """Lance l'acquisition continue (CaptureVideo)."""
        if self._is_live:
            return
        self._check(self._lib.is_CaptureVideo(self._cam, IS_WAIT), "CaptureVideo")
        self._is_live = True

    def stop_live(self):
        """Arrête l'acquisition continue."""
        if not self._is_live:
            return
        self._lib.is_StopLiveVideo(self._cam, IS_WAIT)
        self._is_live = False

    def grab_live(self) -> np.ndarray:
        """
        Lit le dernier frame disponible depuis le buffer live.
        Appeler start_live() au préalable.
        """
        if not self._is_live:
            raise RuntimeError("grab_live() appelé sans start_live()")
        return self._copy_to_numpy()

    # ── Exposition ────────────────────────────────────────────────────────────

    def set_exposure(self, value_ms: float) -> float:
        """
        Fixe l'exposition. Gère automatiquement l'activation/désactivation
        du mode longue exposition si la caméra le supporte.

        La valeur réellement appliquée par le driver (après arrondi à
        l'incrément matériel) est retournée.

        Parameters
        ----------
        value_ms : float
            Exposition souhaitée en millisecondes.

        Returns
        -------
        float
            Exposition réellement appliquée en ms.
        """
        if self.sensor_info is None:
            raise RuntimeError("Caméra non initialisée — appeler open() d'abord")

        # Adaptation du framerate pour laisser la place à l'exposition
        # Le driver sature l'expo à (1/fps - overhead).
        # On impose le fps minimum pour maximiser la plage accessible.
        if value_ms > 1.0:
            self._set_framerate(self.sensor_info.fps_min)

        exp = c_double(value_ms)
        ret = self._lib.is_Exposure(
            self._cam, _EXP_SET, byref(exp), sizeof(exp)
        )
        if ret != IS_SUCCESS:
            raise RuntimeError(f"is_Exposure SET failed: error {ret}")

        return self.get_exposure()

    def get_exposure(self) -> float:
        """Retourne l'exposition courante en ms."""
        val = c_double(0)
        self._lib.is_Exposure(self._cam, _EXP_GET_CURRENT, byref(val), sizeof(val))
        return val.value

    def get_exposure_limits(self) -> Tuple[float, float, float]:
        """
        Retourne (min_ms, max_ms, increment_ms) de l'exposition accessible
        avec le framerate courant.
        """
        return self._get_exposure_range_normal()

    # ── Framerate ─────────────────────────────────────────────────────────────

    def set_framerate(self, fps: float) -> float:
        """
        Fixe le framerate. Retourne le fps réellement appliqué.
        Modifier le fps change implicitement la plage d'exposition accessible.
        """
        return self._set_framerate(fps)

    def get_framerate(self) -> float:
        """Retourne le fps courant mesuré."""
        val = c_double(0)
        self._lib.is_GetFramesPerSecond(self._cam, byref(val))
        return val.value

    # ── Pixel Clock ───────────────────────────────────────────────────────────

    def set_pixelclock(self, mhz: int) -> bool:
        """
        Fixe le pixel clock en MHz.
        Attention : changer le pixel clock modifie les plages de framerate
        et d'exposition — reconfigurer celles-ci après.
        """
        pc_min, pc_max, _ = self._get_pixelclock_range()
        if not (pc_min <= mhz <= pc_max):
            raise ValueError(
                f"Pixel clock {mhz} MHz hors plage [{pc_min}, {pc_max}]"
            )
        val = c_uint32(mhz)
        ret = self._lib.is_PixelClock(self._cam, _PC_SET, byref(val), sizeof(val))
        if self.sensor_info:
            self.sensor_info.pc_current_mhz = mhz
        return ret == IS_SUCCESS

    def get_pixelclock(self) -> int:
        """Retourne le pixel clock courant en MHz."""
        return self._get_pixelclock()

    # ── AOI ───────────────────────────────────────────────────────────────────

    def set_aoi(self, x: int, y: int, width: int, height: int) -> Tuple[int, int, int, int]:
        """
        Définit la région d'intérêt (AOI).
        Réalloue automatiquement le buffer mémoire à la nouvelle taille.

        Returns
        -------
        tuple (x, y, width, height) effectivement appliqués par le driver.
        """
        aoi = _IS_RECT()
        aoi.s32X      = x
        aoi.s32Y      = y
        aoi.s32Width  = width
        aoi.s32Height = height

        ret = self._lib.is_AOI(
            self._cam, IS_AOI_IMAGE_SET_AOI, byref(aoi), sizeof(aoi)
        )
        if ret != IS_SUCCESS:
            raise RuntimeError(f"is_AOI SET failed: error {ret}")

        # Lire l'AOI effective (le driver peut arrondir aux incréments)
        effective = self.get_aoi()
        ew, eh = effective[2], effective[3]

        # Réallouer le buffer si la taille a changé
        if ew != self._width or eh != self._height:
            self._lib.is_FreeImageMem(self._cam, self._mem, self._mem_id)
            self._alloc_mem(ew, eh)

        return effective

    def get_aoi(self) -> Tuple[int, int, int, int]:
        """Retourne (x, y, width, height) de l'AOI courante."""
        aoi = _IS_RECT()
        self._lib.is_AOI(
            self._cam, IS_AOI_IMAGE_GET_AOI, byref(aoi), sizeof(aoi)
        )
        return aoi.s32X, aoi.s32Y, aoi.s32Width, aoi.s32Height

    def reset_aoi(self) -> Tuple[int, int, int, int]:
        """Remet l'AOI au capteur complet."""
        if self.sensor_info is None:
            raise RuntimeError("Caméra non initialisée")
        return self.set_aoi(
            0, 0,
            self.sensor_info.max_width,
            self.sensor_info.max_height
        )

    # ── Propriétés utiles ────────────────────────────────────────────────────

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def is_open(self) -> bool:
        return self._lib is not None and self._cam != 0

    @property
    def is_live(self) -> bool:
        return self._is_live

    # ── Internals ────────────────────────────────────────────────────────────

    def _register_prototypes(self):
        lib = self._lib

        lib.is_InitCamera.argtypes  = [POINTER(c_int), c_void_p]
        lib.is_InitCamera.restype   = c_int

        lib.is_ExitCamera.argtypes  = [c_int]
        lib.is_ExitCamera.restype   = c_int

        lib.is_GetSensorInfo.argtypes = [c_int, c_void_p]
        lib.is_GetSensorInfo.restype  = c_int

        lib.is_SetColorMode.argtypes = [c_int, c_int]
        lib.is_SetColorMode.restype  = c_int

        lib.is_AllocImageMem.argtypes = [
            c_int, c_int, c_int, c_int,
            POINTER(c_char_p), POINTER(c_int)
        ]
        lib.is_AllocImageMem.restype = c_int

        lib.is_SetImageMem.argtypes = [c_int, c_char_p, c_int]
        lib.is_SetImageMem.restype  = c_int

        lib.is_FreeImageMem.argtypes = [c_int, c_char_p, c_int]
        lib.is_FreeImageMem.restype  = c_int

        lib.is_CopyImageMem.argtypes = [c_int, c_char_p, c_int, c_char_p]
        lib.is_CopyImageMem.restype  = c_int

        lib.is_GetImageMemPitch.argtypes = [c_int, POINTER(c_int)]
        lib.is_GetImageMemPitch.restype  = c_int

        lib.is_CaptureVideo.argtypes = [c_int, c_int]
        lib.is_CaptureVideo.restype  = c_int

        lib.is_FreezeVideo.argtypes  = [c_int, c_int]
        lib.is_FreezeVideo.restype   = c_int

        lib.is_StopLiveVideo.argtypes = [c_int, c_int]
        lib.is_StopLiveVideo.restype  = c_int

        lib.is_GetFramesPerSecond.argtypes = [c_int, POINTER(c_double)]
        lib.is_GetFramesPerSecond.restype  = c_int

        lib.is_GetFrameTimeRange.argtypes = [
            c_int, POINTER(c_double), POINTER(c_double), POINTER(c_double)
        ]
        lib.is_GetFrameTimeRange.restype = c_int

        lib.is_SetFrameRate.argtypes = [c_int, c_double, POINTER(c_double)]
        lib.is_SetFrameRate.restype  = c_int

        lib.is_Exposure.argtypes = [c_int, c_uint, c_void_p, c_uint]
        lib.is_Exposure.restype  = c_int

        lib.is_PixelClock.argtypes = [c_int, c_uint, c_void_p, c_uint]
        lib.is_PixelClock.restype  = c_int

        lib.is_SetExternalTrigger.argtypes = [c_int, c_int]
        lib.is_SetExternalTrigger.restype  = c_int

        lib.is_AOI.argtypes = [c_int, c_uint, c_void_p, c_uint]
        lib.is_AOI.restype  = c_int

    def _check(self, ret: int, label: str):
        if ret != IS_SUCCESS:
            raise RuntimeError(f"uc480 [{label}] error code {ret}")

    def _alloc_mem(self, width: int, height: int):
        mem    = c_char_p()
        mem_id = c_int()
        self._check(
            self._lib.is_AllocImageMem(
                self._cam, width, height, 8,
                byref(mem), byref(mem_id)
            ),
            "AllocImageMem"
        )
        self._check(
            self._lib.is_SetImageMem(self._cam, mem, mem_id.value),
            "SetImageMem"
        )
        self._mem    = mem
        self._mem_id = mem_id.value
        self._width  = width
        self._height = height

    def _copy_to_numpy(self) -> np.ndarray:
        """Copie le buffer caméra vers un array numpy C-contiguous uint8."""
        n_bytes = self._width * self._height
        dst = ctypes.create_string_buffer(n_bytes)
        ret = self._lib.is_CopyImageMem(
            self._cam, self._mem, self._mem_id, dst
        )
        if ret != IS_SUCCESS:
            raise RuntimeError(f"is_CopyImageMem failed: error {ret}")
        return np.frombuffer(dst.raw, dtype=np.uint8).reshape(
            (self._height, self._width)
        ).copy()

    def _set_framerate(self, fps: float) -> float:
        new_fps = c_double(0)
        self._lib.is_SetFrameRate(self._cam, c_double(fps), byref(new_fps))
        return new_fps.value

    def _get_fps_range(self) -> Tuple[float, float]:
        ft_min = c_double(); ft_max = c_double(); ft_inc = c_double()
        self._lib.is_GetFrameTimeRange(
            self._cam, byref(ft_min), byref(ft_max), byref(ft_inc)
        )
        fps_min = 1.0 / ft_max.value if ft_max.value > 0 else 1.0
        fps_max = 1.0 / ft_min.value if ft_min.value > 0 else 100.0
        return fps_min, fps_max

    def _exposure_caps(self) -> int:
        caps = c_uint32(0)
        self._lib.is_Exposure(self._cam, _EXP_GET_CAPS, byref(caps), sizeof(caps))
        return caps.value

    def _get_exposure_range_normal(self) -> Tuple[float, float, float]:
        mn  = c_double(0); mx = c_double(0); inc = c_double(0)
        self._lib.is_Exposure(self._cam, _EXP_GET_RANGE_MIN, byref(mn),  sizeof(mn))
        self._lib.is_Exposure(self._cam, _EXP_GET_RANGE_MAX, byref(mx),  sizeof(mx))
        self._lib.is_Exposure(self._cam, _EXP_GET_RANGE_INC, byref(inc), sizeof(inc))
        return mn.value, mx.value, inc.value

    def _get_pixelclock_range(self) -> Tuple[int, int, int]:
        arr = (c_uint32 * 3)(0, 0, 0)
        self._lib.is_PixelClock(self._cam, _PC_GET_RANGE, arr, sizeof(arr))
        return int(arr[0]), int(arr[1]), int(arr[2])

    def _get_pixelclock(self) -> int:
        val = c_uint32(0)
        self._lib.is_PixelClock(self._cam, _PC_GET, byref(val), sizeof(val))
        return int(val.value)

    # ── Context manager ──────────────────────────────────────────────────────

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __repr__(self):
        if self.sensor_info:
            return (
                f"UC480Camera({self.sensor_info.name}, "
                f"{self._width}×{self._height}, "
                f"{'live' if self._is_live else 'idle'})"
            )
        return "UC480Camera(not open)"


# ─────────────────────────────────────────────────────────────────────────────
# Test autonome (sans PyMoDAQ)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    with UC480Camera() as cam:
        info = cam.open()
        print(f"Caméra ouverte : {cam}")
        print(f"  Résolution    : {info.max_width} × {info.max_height}")
        print(f"  Pixel size    : {info.pixel_size_um:.2f} µm")
        print(f"  FPS range     : {info.fps_min:.3f} – {info.fps_max:.1f} Hz")
        print(f"  Expo range    : {info.exp_min_ms:.4f} – {info.exp_max_ms:.3f} ms")
        print(f"  Pixel clock   : {info.pc_current_mhz} MHz [{info.pc_min_mhz}–{info.pc_max_mhz}]")

        for exp_ms in [1, 10, 50, 100, 150]:
            actual = cam.set_exposure(exp_ms)
            arr = cam.snap()
            print(f"  expo={actual:7.3f} ms | shape={arr.shape} | "
                  f"min={arr.min():3d} max={arr.max():3d} mean={arr.mean():.1f}")

        print("\nTest AOI 640×512 @ (320,256)")
        eff = cam.set_aoi(320, 256, 640, 512)
        print(f"  AOI effective : {eff}")
        arr = cam.snap()
        print(f"  Frame shape   : {arr.shape}")

        cam.reset_aoi()
        print(f"  AOI reset → {cam.get_aoi()}")

    print("Caméra fermée proprement.")
