from skmultiflow.drift_detection.base_drift_detector import BaseDriftDetector


class FixedWindowDetector(BaseDriftDetector):

    def __init__(self, window_size):
        super().__init__()
        self.window_size = window_size
        self.current_i = 0

    def reset(self):
        super().reset()
        self.current_i = 0

    def add_element(self, input_value):
        if self.in_concept_change:
            self.reset()

        if self.current_i >= self.window_size:
            self.in_concept_change = True
