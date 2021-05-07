from enum import Enum

class ExplorationSensitivity(Enum):
    linear = 1
    linear_threshold = 2
    linear_iterations = 3
    dynamic_linear = 4

class randomness():
    def __init__(self, r0, rule,threshold=None, sensitivity = 0.999, iterations = None):
        self._randomnesstm1 = r0
        if threshold is not None:
            self._threshold = threshold
        if sensitivity is not None:
            self._sensitivity = sensitivity
        if iterations is not None:
            self._iterations = iterations



        self._update_function = None
        if rule == ExplorationSensitivity.linear:
            self._update_function = self._linear
        elif rule == ExplorationSensitivity.linear_threshold:
            self._update_function = self._linear_threshold
        elif rule == ExplorationSensitivity.linear_iterations:
            self._update_function = self._linear_iterations
        elif rule == ExplorationSensitivity.dynamic_linear:
            self._update_function = self.dynamic_linear
        return

    def next(self,episode,control=None,mov_avg=None,th=None):
        if control is not None:
            self._randomnesstm1 = self._update_function(episode,control=control,mov_avg=mov_avg,th=th)
        else:
            self._randomnesstm1 = self._update_function(episode)
        return self._randomnesstm1

    def _linear(self, episode):
        return (self._sensitivity) * self._randomnesstm1

    def _linear_threshold(self, episode):
        val = self._sensitivity * self._randomnesstm1
        return val if val > self._threshold else self._threshold

    def _linear_iterations(self, episode):
        if episode > self._iterations:
            return self._threshold
        val = self._sensitivity*self._randomnesstm1
        return val

    def dynamic_linear(self,episode,mov_avg,th,control):
        val = self._sensitivity*self._randomnesstm1
        if val > self._threshold:
            if episode < th:
                return val

            else:
                k = -1
                ok = True
                for j in range(control):
                    ok = ok and mov_avg[-j]>=mov_avg[-j-10]
                    if not ok:
                        break

                if ok:
                    return val
                else:
                    return min(0.5,self._sensitivity*1.5)
        else:
            return self._threshold







