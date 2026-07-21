import traceback
import numpy as np

from pymodaq.control_modules.viewer_utility_classes import (
    comon_parameters,
    main
)

from pylablib.devices import uc480

from pymodaq_plugins_utils.hardware.camera_base_pylablib import (
    CameraBasePyLabLib,
    cam_params
)



class DAQ_2DViewer_UC480(CameraBasePyLabLib):

    """
    PyMoDAQ plugin for Thorlabs UC480 cameras
    """



    serial_numbers = [
        cam.serial_number
        for cam in uc480.list_cameras()
    ]


    serial_params = [
        {
            'title': 'Serial number:',
            'name': 'serial_number',
            'type': 'list',
            'limits': serial_numbers
        }
    ]


    params = comon_parameters + serial_params + cam_params



    # ----------------------------------------------------
    # INIT
    # ----------------------------------------------------

    def ini_attributes(self):

        print("UC480: ini_attributes")

        super().ini_attributes()

        self.controller = None



    def ini_detector_custom(self, controller=None):

        print("UC480: init detector")


        if self.is_master:


            serial = self.settings.child(
                'serial_number'
            ).value()


            print(
                "UC480 serial:",
                serial
            )


            dev_id = uc480.UC480Camera.find_by_serial(
                serial
            )


            self.controller = uc480.UC480Camera(
                dev_id=dev_id
            )


        else:

            self.controller = controller



        print(
            "UC480 camera ready"
        )


        self.install_buffer_patch()



    # ----------------------------------------------------
    # PATCH pylablib
    # ----------------------------------------------------

    def install_buffer_patch(self):

        """
        Prevent pylablib crash when UC480 buffer is empty
        """


        camera = self.controller


        original = (
            camera.read_multiple_images
        )


        def safe_read_multiple_images(*args, **kwargs):

            try:

                return original(
                    *args,
                    **kwargs
                )


            except ValueError as e:


                if (
                    "need at least one array"
                    in str(e)
                ):

                    print(
                        "UC480: empty buffer"
                    )


                    return (
                        np.empty(
                            (0,),
                            dtype=np.uint16
                        ),
                        (0,0)
                    )


                raise



        camera.read_multiple_images = (
            safe_read_multiple_images
        )


        print(
            "UC480 buffer patch installed"
        )



    # ----------------------------------------------------
    # SETTINGS
    # ----------------------------------------------------

    def commit_settings(self,param):


        print(
            "UC480 commit:",
            param.name(),
            param.value()
        )


        if self.controller is None:

            super().commit_settings(param)
            return



        try:


            with self.controller.pausing_acquisition():


                super().commit_settings(param)



                if param.name() == "exposure_time":


                    exposure = (
                        param.value()
                        /
                        1000
                    )


                    print(
                        "UC480 exposure:",
                        exposure,
                        "s"
                    )


                    #
                    # IMPORTANT
                    # ne pas toucher au frame period
                    #

                    self.controller.set_exposure(
                        exposure
                    )



        except Exception:


            traceback.print_exc()



    # ----------------------------------------------------
    # ACQUISITION
    # ----------------------------------------------------


    def start_acquisition(self):

        print(
            "UC480 start acquisition"
        )


        try:

            self.controller.start_acquisition()


        except Exception:


            traceback.print_exc()



    def stop_acquisition(self):

        print(
            "UC480 stop acquisition"
        )


        try:

            self.controller.stop_acquisition()


        except Exception as e:

            print(
                "stop acquisition:",
                e
            )



        try:

            self.controller.clear_buffers()


            print(
                "UC480 buffers cleared"
            )


        except Exception as e:

            print(
                "clear buffers:",
                e
            )



    # ----------------------------------------------------
    # GRAB
    # ----------------------------------------------------

    def grab_data(
            self,
            Naverage=1,
            **kwargs):


        print(
            "UC480 grab"
        )


        try:


            return super().grab_data(
                Naverage,
                **kwargs
            )


        except Exception as e:


            print(
                "UC480 grab error:",
                e
            )


            traceback.print_exc()


            return None



    # ----------------------------------------------------
    # CLOSE
    # ----------------------------------------------------

    def close(self):


        print(
            "UC480 close"
        )


        if self.controller is not None:


            try:

                self.controller.stop_acquisition()


            except Exception:

                pass



            try:

                self.controller.clear_buffers()


            except Exception:

                pass



            try:

                self.controller.close()


            except Exception:


                traceback.print_exc()



        print(
            "UC480 closed"
        )





if __name__ == '__main__':

    main(
        __file__,
        init=False
    )