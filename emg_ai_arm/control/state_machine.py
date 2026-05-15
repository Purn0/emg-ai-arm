class Controller:
    """
    Mode-based controller for EMG-driven robotic arm

    Modes:
    0 -> GRIP
    1 -> WRIST
    2 -> ARM
    """

    def __init__(self):
        self.mode = 0
        self.mode_names = ["GRIP", "WRIST", "ARM"]
        self.hold_counter = 0

        # Tunable parameters
        self.switch_strength_thresh = 0.45   # tolerant for real EMG
        self.switch_windows_required = 3     # 3 * 200ms = ~0.6s

        # Cooldown to prevent rapid re-switching
        self.cooldown = 0
        self.cooldown_windows = 6            # 6 * 200ms = ~1.2s

    def update(self, pred: int, strength: float) -> str:
        """
        pred:
            0 = REST
            1 = CH1
            2 = CH2
            3 = BOTH (mode switch)

        strength:
            activation level (0.0 - 1.0 approx)
        """

        # =========================
        # COOLDOWN LOGIC
        # =========================
        if self.cooldown > 0:
            self.cooldown -= 1
            # During cooldown, ignore BOTH entirely
            if pred == 3:
                return "BOTH_IGNORED (cooldown)"
        # =========================
        # MODE SWITCH LOGIC
        # =========================
        if pred == 3 and strength >= self.switch_strength_thresh:
            self.hold_counter += 1
        else:
            self.hold_counter = 0

        if self.hold_counter >= self.switch_windows_required:
            self.mode = (self.mode + 1) % 3
            self.hold_counter = 0
            self.cooldown = self.cooldown_windows
            return f"MODE_SWITCH -> {self.mode_names[self.mode]}"

        if pred == 3:
            return "HOLD_BOTH (switching...)"

        # =========================
        # REST
        # =========================
        if pred == 0:
            return "REST"

        # =========================
        # MODE-SPECIFIC CONTROL
        # =========================
        speed = max(0.0, min(strength, 1.0))  # clamp safety

        if self.mode == 0:  # GRIP
            if pred == 1:
                return f"GRIP_CLOSE speed={speed:.2f}"
            if pred == 2:
                return f"GRIP_OPEN  speed={speed:.2f}"

        if self.mode == 1:  # WRIST
            if pred == 1:
                return f"WRIST_LEFT  speed={speed:.2f}"
            if pred == 2:
                return f"WRIST_RIGHT speed={speed:.2f}"

        if self.mode == 2:  # ARM
            if pred == 1:
                return f"ARM_UP   speed={speed:.2f}"
            if pred == 2:
                return f"ARM_DOWN speed={speed:.2f}"

        return "UNKNOWN"
