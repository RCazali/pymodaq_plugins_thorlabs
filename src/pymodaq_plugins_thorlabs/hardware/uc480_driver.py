import ctypes
from ctypes import (
    Structure, POINTER, byref, create_string_buffer, sizeof,
    c_int, c_uint, c_uint32, c_char, c_void_p, c_double
)
import numpy as np
import cv2

IS_USE_FIRST_AVAILABLE_CAMERA = 0
IS_USE_CAMERA_ID = 0x8000

# Mode monochrome direct 8-bit (1 octet/pixel)
IS_CM_MONO8 = 6

IS_AOI_IMAGE_SET_AOI = 0x0001
IS_SUCCESS = 0

class IS_RECT(Structure):
    _fields_ = [
        ("s32X", c_int), ("s32Y", c_int),
        ("s32Width", c_int), ("s32Height", c_int),
    ]

class IS_SENSOR_INFO(Structure):
    _fields_ = [
        ("SensorID", ctypes.c_uint16),
        ("strSensorName", c_char * 32),
        ("nColorMode", c_char),
        ("nMaxWidth", c_uint),
        ("nMaxHeight", c_uint),
        ("bMasterGain", ctypes.c_bool),
        ("bRGain", ctypes.c_bool),
        ("bGGain", ctypes.c_bool),
        ("bBGain", ctypes.c_bool),
        ("bGlobShutter", ctypes.c_bool),
        ("Reserved", c_char * 16),
    ]

class UC480Camera:
    def __init__(self, dll_name: str = "uc480_64.dll"):
        self._lib = ctypes.WinDLL(dll_name)
        self._cam = c_int(0)
        self._mem = None
        self._mem_id = c_int(0)
        self._width = 0
        self._height = 0
        self._pitch = 0

        self._register_prototypes()

    def _register_prototypes(self):
        lib = self._lib
        lib.is_InitCamera.argtypes = [POINTER(c_int), c_void_p]
        lib.is_InitCamera.restype = c_int

        lib.is_ExitCamera.argtypes = [c_int]
        lib.is_ExitCamera.restype = c_int

        lib.is_GetSensorInfo.argtypes = [c_int, POINTER(IS_SENSOR_INFO)]
        lib.is_GetSensorInfo.restype = c_int

        lib.is_AllocImageMem.argtypes = [c_int, c_int, c_int, c_int, POINTER(c_void_p), POINTER(c_int)]
        lib.is_AllocImageMem.restype = c_int

        lib.is_FreeImageMem.argtypes = [c_int, c_void_p, c_int]
        lib.is_FreeImageMem.restype = c_int

        lib.is_SetImageMem.argtypes = [c_int, c_void_p, c_int]
        lib.is_SetImageMem.restype = c_int

        lib.is_SetColorMode.argtypes = [c_int, c_int]
        lib.is_SetColorMode.restype = c_int

        lib.is_AOI.argtypes = [c_int, c_uint, c_void_p, c_uint]
        lib.is_AOI.restype = c_int

        lib.is_FreezeVideo.argtypes = [c_int, c_int]
        lib.is_FreezeVideo.restype = c_int

        lib.is_CopyImageMem.argtypes = [c_int, c_void_p, c_int, c_void_p]
        lib.is_CopyImageMem.restype = c_int

        lib.is_SetExposureTime.argtypes = [c_int, c_double, POINTER(c_double)]
        lib.is_SetExposureTime.restype = c_int

        lib.is_SetFrameRate.argtypes = [c_int, c_double, POINTER(c_double)]
        lib.is_SetFrameRate.restype = c_int

        lib.is_PixelClock.argtypes = [c_int, c_uint, c_void_p, c_uint]
        lib.is_PixelClock.restype = c_int

        lib.is_SetHardwareGain.argtypes = [c_int, c_int, c_int, c_int, c_int]
        lib.is_SetHardwareGain.restype = c_int

        lib.is_ResetToDefault.argtypes = [c_int]
        lib.is_ResetToDefault.restype = c_int

        lib.is_GetImageMemPitch.argtypes = [c_int, POINTER(c_int)]
        lib.is_GetImageMemPitch.restype = c_int

    def open(self, camera_id: int = 2):
        h_cam = c_int(camera_id | IS_USE_CAMERA_ID if camera_id != 0 else IS_USE_FIRST_AVAILABLE_CAMERA)

        ret = self._lib.is_InitCamera(byref(h_cam), None)
        if ret != IS_SUCCESS:
            raise RuntimeError(f"Échec init (Code {ret})")

        self._cam = h_cam.value

        # Reset usine
        self._lib.is_ResetToDefault(self._cam)

        # 1. Pixel clock minimal (10 MHz)
        pclk = c_uint(10)
        self._lib.is_PixelClock(self._cam, 1, byref(pclk), sizeof(pclk))

        # 2. FPS à 1.0 FPS pour débloquer les 500ms d'exposition
        act_fr = c_double(0.0)
        self._lib.is_SetFrameRate(self._cam, c_double(1.0), byref(act_fr))

        # 3. Augmenter le gain matériel pour régler le problème d'image sombre (Master Gain à 50%)
        # Paramètres : nMaster (0..100), nRed, nGreen, nBlue
        self._lib.is_SetHardwareGain(self._cam, c_int(50), c_int(-1), c_int(-1), c_int(-1))

        # 4. Informations capteur
        sensor_info = IS_SENSOR_INFO()
        self._lib.is_GetSensorInfo(self._cam, byref(sensor_info))
        
        max_w = sensor_info.nMaxWidth
        max_h = sensor_info.nMaxHeight

        # AOI Plein capteur
        rect = IS_RECT(0, 0, max_w, max_h)
        self._lib.is_AOI(self._cam, IS_AOI_IMAGE_SET_AOI, byref(rect), sizeof(rect))

        self._width = max_w
        self._height = max_h

        # 5. Mode MONO8 strict
        self._lib.is_SetColorMode(self._cam, IS_CM_MONO8)

        # Allocation mémoire 8 bits
        self._alloc_mem(self._width, self._height, bpp=8)

    def _alloc_mem(self, width: int, height: int, bpp: int = 8):
        if self._mem is not None:
            self._free_mem()

        mem_ptr = c_void_p()
        mem_id = c_int(0)

        ret = self._lib.is_AllocImageMem(self._cam, width, height, bpp, byref(mem_ptr), byref(mem_id))
        if ret != IS_SUCCESS:
            raise RuntimeError(f"Échec Alloc (Code {ret})")

        self._lib.is_SetImageMem(self._cam, mem_ptr, mem_id)
        self._mem = mem_ptr
        self._mem_id = mem_id

        pitch_c = c_int(0)
        self._lib.is_GetImageMemPitch(self._cam, byref(pitch_c))
        self._pitch = pitch_c.value

    def _free_mem(self):
        if self._mem is not None and self._mem_id.value != 0:
            self._lib.is_FreeImageMem(self._cam, self._mem, self._mem_id)
            self._mem = None
            self._mem_id = c_int(0)

    def set_exposure(self, exposure_ms: float) -> float:
        act_exp = c_double(0.0)
        self._lib.is_SetExposureTime(self._cam, c_double(exposure_ms), byref(act_exp))
        return act_exp.value

    def capture_image(self, timeout_ms: int = 5000) -> np.ndarray:
        self._lib.is_SetImageMem(self._cam, self._mem, self._mem_id)

        ret = self._lib.is_FreezeVideo(self._cam, c_int(timeout_ms))
        if ret != IS_SUCCESS:
            raise RuntimeError(f"Échec FreezeVideo (Code {ret})")

        buffer_size = self._pitch * self._height
        buffer = create_string_buffer(buffer_size)

        ret_copy = self._lib.is_CopyImageMem(self._cam, self._mem, self._mem_id, buffer)
        if ret_copy != IS_SUCCESS:
            raise RuntimeError(f"Échec CopyImageMem (Code {ret_copy})")

        raw_flat = np.frombuffer(buffer, dtype=np.uint8)
        
        # Reshape direct 2D grâce à la stricte concordance MONO8
        raw_2d = raw_flat.reshape((self._height, self._pitch))
        
        # Tranchage sur la largeur utile
        clean_img = raw_2d[:, :self._width]

        return clean_img

    def close(self):
        if self._cam != 0:
            self._free_mem()
            self._lib.is_ExitCamera(self._cam)
            self._cam = 0


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    print("==================================================")
    print("      MODE MONO8 + AUTO-SCALING + VIRIDIS        ")
    print("==================================================")

    cam = UC480Camera()
    try:
        cam.open(camera_id=2)
        
        exp_reelle = cam.set_exposure(500.0)
        img_mono = cam.capture_image(timeout_ms=5000)

        print(f"  * Resolution physique : {cam._width} x {cam._height}")
        print(f"  * Exposition reelle   : {exp_reelle:.2f} ms")
        print(f"  * Niveaux min / max   : {img_mono.min()} / {img_mono.max()}")

        # Affichage avec Colormap Viridis et contraste ajusté
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # vmin/vmax dynamiques pour rendre le signal bien visible
        im = ax.imshow(img_mono, cmap='viridis', vmin=img_mono.min(), vmax=max(img_mono.max(), 1))
        ax.set_title(f"Image Cadrée Plein Champ ({img_mono.shape[1]}x{img_mono.shape[0]})")
        
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Intensité (0-255)')

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"\n[ERREUR] : {e}")

    finally:
        cam.close()
        print("\nCaméra fermée.")