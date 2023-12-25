from pydualsense import *
import pyautogui
import math


class ps5ControllerInterface:

    def __init__(self, multiple_monitors: bool = False):
        print("Starting...")
        # Get controller
        self.dualsense = pydualsense()
        self.dualsense.init()
        self.dualsense.light.setColorI(160, 70, 255)
        # Resolution and monitor handling
        self.monitor = 0
        self.buttonPressed = False
        self.multiple_monitors = multiple_monitors
        self.resolution = pyautogui.size()
        self.offset = (0, 0)
        self.ratio = (self.resolution[0] / 1920, self.resolution[1] / 1080)
        self.permanentOffset = (0, 0)
        self.aiming = False
        pyautogui.FAILSAFE = False

        # Mouse clicks:
        self.dualsense.r1_changed.subscribe(self.leftClick)
        self.dualsense.touch_pressed.subscribe(self.leftClick)
        self.dualsense.l1_changed.subscribe(self.rightClick)
        self.dualsense.l3_changed.subscribe(self.pullOutKeyboard)
        self.dualsense.r3_changed.subscribe(self.recenter)
        # Arrow keys:
        self.dualsense.dpad_left.subscribe(self.left)
        self.dualsense.dpad_right.subscribe(self.right)
        self.dualsense.dpad_up.subscribe(self.up)
        self.dualsense.dpad_down.subscribe(self.down)
        # Special keys:
        self.dualsense.square_pressed.subscribe(self.backspace)
        self.dualsense.circle_pressed.subscribe(self.escape)
        self.dualsense.r2_changed.subscribe(self.shift)
        self.dualsense.l2_changed.subscribe(self.caps)
        self.dualsense.triangle_pressed.subscribe(self.space)
        self.dualsense.cross_pressed.subscribe(self.leftClick)


        # Auto-loop:
        self.trackingLoop() # Break out of this with the share key

    def trackingLoop(self):
        while not self.dualsense.state.share:
            trackpad = self.dualsense.state.trackPadTouch0
            if trackpad.isActive:
                if not self.aiming:
                    self.aiming = True
                    self.offset = (trackpad.X, trackpad.Y)
                    self.permanentOffset = pyautogui.position()

                x = math.floor((trackpad.X - self.offset[0]) * self.ratio[0]) + self.permanentOffset[0]
                y = math.floor((trackpad.Y - self.offset[1]) * self.ratio[1]) + self.permanentOffset[1]
                pyautogui.moveTo(x, y, _pause=False)
            else:
                self.aiming = False
        self.stop()


    # CONSIDER RUNNING AN ALPHANUMERIC RECOGNIZER ON WHATEVER SOMEONE DRAWS ON THE TOUCHPAD
    def pullOutKeyboard(self, state = None):
            if self.dualsense.state.L3:
                pyautogui.hotkey('alt', 'shift', 'o') # Shortcut for pulling our Google's on-screen keyboard
                # pyautogui.hotkey('win', 'ctrl', 'o') # For Windows, currently broken
    
    def recenter(self, state = None):
        pyautogui.moveTo(self.resolution[0] / 2, self.resolution[1] / 2, _pause=False)

    # Mouse Clicks:
    def leftClick(self, state = None):
        if self.dualsense.state.R1 or self.dualsense.state.touchBtn or self.dualsense.state.cross:
            pyautogui.mouseDown(button='left')
        else:
            pyautogui.mouseUp(button='left')
    def rightClick(self, state = None):
        if self.dualsense.state.L1:
            pyautogui.mouseDown(button='right')
        else:
            pyautogui.mouseUp(button='right')

    # Arrow Keys:
    def left(self, state = None):
        if self.dualsense.state.DpadLeft:
            pyautogui.keyDown('left')
        else:
            pyautogui.keyUp('left')
    def right(self, state = None):
        if self.dualsense.state.DpadRight:
            pyautogui.keyDown('right')
        else:
            pyautogui.keyUp('right')
    def up(self, state = None):
        if self.dualsense.state.DpadUp:
            pyautogui.keyDown('up')
        else:
            pyautogui.keyUp('up')
    def down(self, state = None):
        if self.dualsense.state.DpadDown:
            pyautogui.keyDown('down')
        else:
            pyautogui.keyUp('down')

    # Special Keys (square, circle, triangle):
    def backspace(self, state = None):
        if self.dualsense.state.square:
            pyautogui.keyDown('backspace')
        else:
            pyautogui.keyUp('backspace')
    def escape(self, state = None):
        if self.dualsense.state.circle:
            pyautogui.keyDown('esc')
        else:
            pyautogui.keyUp('esc')
    def shift(self, state = None):
        if self.dualsense.state.R2Btn:
            pyautogui.keyDown('shift')
        else:
            pyautogui.keyUp('shift')
    def space(self, state = None):
        if self.dualsense.state.triangle:
            pyautogui.keyDown('space')
        else:
            pyautogui.keyUp('space')
    def caps(self, state = None):
        if self.dualsense.state.L2Btn:
            pyautogui.keyDown('capslock')
        else:
            pyautogui.keyUp('capslock')


    def stop(self, state = None):
        self.dualsense.close()
        print("Ending...")



if __name__ == "__main__":
    gyroReader = ps5ControllerInterface(multiple_monitors = True)
