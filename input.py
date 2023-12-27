from pydualsense import *
import math
import textRecognition as tr
import sys
import time
import mouse # TODO: REPLACE THIS WITH pyinput.mouse (Controller)
from pynput.keyboard import Key, Controller as KeyboardController


class ps5ControllerInterface:

    def __init__(self, resolution = (2560, 1440)):
        print("Starting...")
        # Get controller
        self.dualsense = pydualsense()
        self.dualsense.init()
        self.dualsense.light.setColorI(160, 70, 255)
        # Resolution and monitor handling
        self.monitor = 0
        self.buttonPressed = False
        self.resolution = resolution
        self.offset = (0, 0)
        self.ratio = (self.resolution[0] / 1920, self.resolution[1] / 1080)
        self.permanentOffset = (0, 0)
        self.aiming = False
        # Keyboard:
        self.keyboard = KeyboardController()

        # Model:
        self.ps_pressed = 0
        self.model = tr.TextRecognition()
        self.dualsense.ps_pressed.subscribe(self.activateModel)

        # Auto-loop:
        self.addDefaultListeners() # Break out of this with the share key

    def addDefaultListeners(self):
        # Model Handling:
        self.dualsense.ps_pressed.subscribe(self.activateModel)
        self.dualsense.trackpad_frame_reported.subscribe(self.trackpadOn)
        self.dualsense.trackpad_off.subscribe(self.trackpadOff)
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
        # stoppage:
        self.dualsense.share_pressed.subscribe(self.stop)


    def trackpadOn(self, state = None):
        trackpad = self.dualsense.state.trackPadTouch0
        # MOUSE HANDLING: (+ TRACKPAD | - MODEL | ? TRACKING FOR MODEL)
        if not self.model.isActive:
            if not self.aiming:
                self.aiming = True
                self.offset = (trackpad.X, trackpad.Y)
                self.permanentOffset = mouse.get_position()

            x = math.floor((trackpad.X - self.offset[0]) * self.ratio[0]) + self.permanentOffset[0]
            y = math.floor((trackpad.Y - self.offset[1]) * self.ratio[1]) + self.permanentOffset[1]
            mouse.move(x, y, absolute=True, duration=0.0)
        
        # MODEL TRACKING: (+ TRACKPAD | + MODEL | [+] TRACKING FOR MODEL)
        elif self.model.isActive:
            self.model.tracking = True
            self.model.addPoint((trackpad.X, trackpad.Y))

    def trackpadOff(self, state = None):
        if self.model.isActive and self.model.tracking:
                # Predict:
                self.model.matrixFromPositions()
                self.model.classifier.predict(self.model.matrix)
                self.model.savePNG(self.model.matrix) #* Remove in final version?
                self.model.tracking = False
        else:
            self.aiming = False

    def trackingLoop(self):
        while not self.dualsense.state.share:
            trackpad = self.dualsense.state.trackPadTouch0
            if trackpad.isActive:
                # MOUSE HANDLING: (+ TRACKPAD | - MODEL | ? TRACKING FOR MODEL)
                if not self.model.isActive:
                    # Set our initial position:
                    if not self.aiming:
                        self.aiming = True
                        self.offset = (trackpad.X, trackpad.Y)
                        self.permanentOffset = mouse.get_position()

                    x = math.floor((trackpad.X - self.offset[0]) * self.ratio[0]) + self.permanentOffset[0]
                    y = math.floor((trackpad.Y - self.offset[1]) * self.ratio[1]) + self.permanentOffset[1]
                    mouse.move(x, y, absolute=True, duration=0.0)
                
                # MODEL TRACKING: (+ TRACKPAD | + MODEL | [+] TRACKING FOR MODEL)
                elif self.model.isActive:
                    self.model.tracking = True
                    self.model.addPoint((trackpad.X, trackpad.Y))
            
            else:
                self.trackpadOff()

        self.stop()

    def activateModel(self, state = None):
        self.ps_pressed = (self.ps_pressed + 1) % 4 # IDK man, I'm just a little guy...

        if self.ps_pressed == 1:
            self.model.isActive = not self.model.isActive
            if self.model.isActive:
                print("Model mode")
                self.dualsense.light.setColorI(255, 0, 0)
            else:
                print("Mouse mode")
                self.dualsense.light.setColorI(160, 70, 255)

    # CONSIDER RUNNING AN ALPHANUMERIC RECOGNIZER ON WHATEVER SOMEONE DRAWS ON THE TOUCHPAD
    def pullOutKeyboard(self, state = None):
            if self.dualsense.state.L3:
                hotkey_combination = [Key.cmd, Key.ctrl, 'o']
                with self.keyboard.pressed(hotkey_combination[0]):
                    with self.keyboard.pressed(hotkey_combination[1]):
                        self.keyboard.press(hotkey_combination[2])
                        self.keyboard.release(hotkey_combination[2]) 
                
    
    def recenter(self, state = None):
        mouse.move(self.resolution[0] / 2, self.resolution[1] / 2, absolute=True, duration=0.0)

    # Mouse Clicks:
    def leftClick(self, state = None):
        if self.dualsense.state.R1 or self.dualsense.state.touchBtn or self.dualsense.state.cross:
            mouse.press(button='left')
        else:
            pass
            mouse.release(button='left')
    def rightClick(self, state = None):
        if self.dualsense.state.L1:
            mouse.press(button='right')
        else:
            mouse.release(button='right')

    # Arrow Keys:
    def left(self, state = None):
        if self.dualsense.state.DpadLeft:
            self.keyboard.press(Key.left)
        else:
            self.keyboard.release(Key.left)
    def right(self, state = None):
        if self.dualsense.state.DpadRight:
            self.keyboard.press(Key.right)
        else:
            self.keyboard.release(Key.right)
    def up(self, state = None):
        if self.dualsense.state.DpadUp:
            self.keyboard.press(Key.up)
        else:
            self.keyboard.release(Key.up)
    def down(self, state = None):
        if self.dualsense.state.DpadDown:
            self.keyboard.press(Key.down)
        else:
            self.keyboard.release(Key.down)

    # Special Keys (square, circle, triangle):
    def backspace(self, state = None):
        if self.dualsense.state.square:
            self.keyboard.press(Key.backspace)
        else:
            self.keyboard.release(Key.backspace)
    def escape(self, state = None):
        if self.dualsense.state.circle:
            self.keyboard.press(Key.esc)
        else:
            self.keyboard.release(Key.esc)
    def shift(self, state = None):
        if self.dualsense.state.R2Btn:
            self.keyboard.press(Key.shift)
        else:
            self.keyboard.release(Key.shift)
    def space(self, state = None):
        if self.dualsense.state.triangle:
            self.keyboard.press(Key.space)
        else:
            self.keyboard.release(Key.space)
    def caps(self, state = None):
        if self.dualsense.state.L2Btn:
            self.keyboard.press(Key.caps_lock)
        else:
            self.keyboard.release(Key.caps_lock)


    def stop(self, state = None):
        self.dualsense.light.setColorI(160, 70, 255)
        print("Ending...")
        self.dualsense.close()



if __name__ == "__main__":
    gyroReader = ps5ControllerInterface(resolution=(2560, 1440))
