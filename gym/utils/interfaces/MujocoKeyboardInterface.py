"""Keyboard teleop for the MuJoCo passive viewer.

Wires into mujoco.viewer.launch_passive's key_callback, set on the backend
before the first render() call.  MuJoCo delivers discrete key EVENTS, so a
press maps straight to one command increment (no edge detection needed —
contrast VsimKeyboardInterface, which polls).

Bindings and command logic are shared with the vsim interface — see
teleop_bindings.py for the key layout and why it is IJKL/NM.
"""

from gym.utils.interfaces.teleop_bindings import KEY_TO_ACTION, TeleopCommands

# GLFW uses ASCII codes for letters and separate codes for special keys.
# Hardcoded to avoid a glfw dependency.
SPECIAL_KEYCODES = {"UP": 265, "DOWN": 264}
KEYCODE_TO_ACTION = {
    ord(key) if len(key) == 1 else SPECIAL_KEYCODES[key]: action
    for key, action in KEY_TO_ACTION.items()
}
# Viewer-only binding, outside the shared teleop table: toggles the camera
# lock on the robot (MuJoCoCPUBackend.set_camera_tracking). It is not a
# command action, so the vsim interface has no counterpart.
CAMERA_FOLLOW_KEYCODE = ord("F")


class MujocoKeyboardInterface:
    def __init__(self, env):
        self.env = env
        self.commands = TeleopCommands(env)

        env._backend._viewer_key_callback = self._on_key
        self.commands.print_help("MuJoCo viewer")
        print("  F             toggle camera lock on the robot")

    def _on_key(self, keycode: int) -> None:
        if keycode == CAMERA_FOLLOW_KEYCODE:
            backend = self.env._backend
            if not backend.supports_camera_tracking:
                print("camera lock needs a floating-base robot")
                return
            backend.set_camera_tracking(not backend.camera_tracking)
            return
        action = KEYCODE_TO_ACTION.get(keycode)
        if action is not None:
            self.commands.apply(action)
