import pyautogui

import time

from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

pyautogui.FAILSAFE = False

w, h = pyautogui.size()

devices = AudioUtilities.GetSpeakers()

interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)

volume = cast(interface, POINTER(IAudioEndpointVolume))

volRange = volume.GetVolumeRange()

minVol = volRange[0]

maxVol = volRange[1]

vol = 0

while True:

    x, y = pyautogui.position()

    volBar = int(pyautogui.mapRange(y, 0, h, 400, 150))

    volPer = int(pyautogui.mapRange(y, 0, h, 0, 100))

    

    if pyautogui.isPressed('left'):

        vol = max(vol + 2, minVol)

    elif pyautogui.isPressed('right'):

        vol = min(vol - 2, maxVol)

        

    volume.SetMasterVolumeLevel(vol, None)

    

    pyautogui.draw.rect(pyautogui.screenshot(),

                        (50, 150, 35, 250),

                        (255, 0, 0),

                        3)

    pyautogui.draw.rect(pyautogui.screenshot(),

                        (50, volBar, 35, 250 - volBar),

                        (255, 0, 0),

                        -1)

    pyautogui.write(f'{volPer}%', font=('arial', 16, 'bold'),

                    position=(20, 450), color='red')

    

    time.sleep(0.01)

