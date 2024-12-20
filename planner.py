from typing import List, Tuple
from enum import Enum, auto

import matplotlib.pyplot as plt
import matplotlib.patches as mplpatches

debugPrintEnabled = True
useTorqueHeadroom = True


def debugPrint(string: str) -> None:
    if debugPrintEnabled:
        print(string)


def rpmToRps(rpm: float) -> float:
    return rpm / 60


def mNMToNM(mNM: float) -> float:
    return mNM / 1000


class State:
    def __init__(self, positionRot: float, velocityRotPs: float) -> None:
        self.positionRot = positionRot
        self.velocityRotPs = velocityRotPs


simTimestepPeriodSeconds: float = (1 / 1000) / 1000  # 1 us

goalRotations: float = 0.5

maxRotPs: float = rpmToRps(700)

torqueCurve: List[Tuple[float, float]] = [
    (0, 0.42),
    (rpmToRps(60), mNMToNM(320)),
    (rpmToRps(120), mNMToNM(330)),
    (rpmToRps(180), mNMToNM(340)),
    (rpmToRps(240), mNMToNM(350)),
    (rpmToRps(300), mNMToNM(360)),
    (rpmToRps(360), mNMToNM(355)),
    (rpmToRps(420), mNMToNM(345)),
    (rpmToRps(480), mNMToNM(320)),
    (rpmToRps(540), mNMToNM(300)),
    (rpmToRps(600), mNMToNM(275)),
    (rpmToRps(660), mNMToNM(250)),
    (rpmToRps(720), mNMToNM(235)),
]

torqueHeadroom = 0.25

systemMomentKgMSq = 0.0002  # TODO get actual value


def interpolateTorque(rpsKey: float, useHeadroom: bool) -> float:
    if rpsKey < torqueCurve[0][0] or rpsKey > torqueCurve[-1][0]:
        debugPrint("rpsKey out of range!")
        return 0

    mult = (1 - torqueHeadroom) if useHeadroom else 1

    lowerBoundIndex: int = 0

    for i in range(0, len(torqueCurve)):  # pylint: disable=C0200
        if torqueCurve[i][0] == rpsKey:
            return mult * torqueCurve[i][1]
        if torqueCurve[i][0] < rpsKey:
            lowerBoundIndex = i

    lowerBound = torqueCurve[lowerBoundIndex]
    upperBound = torqueCurve[lowerBoundIndex + 1]

    # proportion between upper and lower bounds (0,1)
    intersection: float = (rpsKey - lowerBound[0]) / (upperBound[0] - lowerBound[0])

    return mult * (
        (upperBound[1] * intersection) + (lowerBound[1] * (1 - intersection))
    )


positionSeries: List[float] = []
velocitySeries: List[float] = []
accelSeries: List[float] = []
timeSeries: List[float] = []


def resetGraph():
    positionSeries.clear()
    velocitySeries.clear()
    accelSeries.clear()
    timeSeries.clear()


class PhaseType(Enum):
    FORWARD = auto()
    REVERSE = auto()
    COAST = auto()


def timestep(inputState: State, phaseType: PhaseType, applyMaxVLimit: bool) -> State:
    angAccel = (
        interpolateTorque(abs(inputState.velocityRotPs), useTorqueHeadroom)
        / systemMomentKgMSq
    )

    if phaseType == PhaseType.REVERSE:
        angAccel = -angAccel
    elif phaseType == PhaseType.COAST:
        angAccel = 0

    if applyMaxVLimit:
        if angAccel > 0 and inputState.velocityRotPs > maxRotPs:
            angAccel = 0
        elif angAccel < 0 and -inputState.velocityRotPs > maxRotPs:
            angAccel = 0

    angPos = (
        inputState.positionRot
        + (inputState.velocityRotPs * simTimestepPeriodSeconds)
        + (0.5 * angAccel * simTimestepPeriodSeconds * simTimestepPeriodSeconds)
    )
    angVel = inputState.velocityRotPs + (angAccel * simTimestepPeriodSeconds)

    positionSeries.append(angPos)
    velocitySeries.append(angVel)
    accelSeries.append(angAccel)

    return State(angPos, angVel)


velocityTolerance = 0.01
positionTolerance = 0.001

positionIterationStepSizeS = simTimestepPeriodSeconds * 100

state = State(0, 0)


def resetState(toReset: State):
    toReset.positionRot = 0
    toReset.velocityRotPs = 0


param_accelTimeS = (
    0.04  # held constant (although negative coast times act as cutting into this phase)
)

param_coastTimeS = 0.015  # initial value, iterated to actual working value

simTimeS: float = 0

lastPosition: float = 0

# world's shittiest gradient descent
while abs(lastPosition - goalRotations) > positionTolerance:
    resetGraph()
    resetState(state)
    simTimeS = 0

    phase = PhaseType.FORWARD

    while not (
        phase == PhaseType.REVERSE and abs(state.velocityRotPs) < velocityTolerance
    ):
        if param_coastTimeS >= 0:
            if simTimeS < param_accelTimeS:
                phase = PhaseType.FORWARD
            elif simTimeS > param_accelTimeS + param_coastTimeS:
                phase = PhaseType.REVERSE
            else:
                phase = PhaseType.COAST
        else:
            if simTimeS < param_accelTimeS + param_coastTimeS:
                phase = PhaseType.FORWARD
            else:
                phase = PhaseType.REVERSE

        state = timestep(state, phase, True)

        simTimeS += simTimestepPeriodSeconds

        timeSeries.append(simTimeS * 1000)

    lastPosition = state.positionRot
    debugPrint(
        "rotation state: "
        + str(state.positionRot)
        + ", coast time param: "
        + str(param_coastTimeS * 1000)
        + " ms"
    )

    if state.positionRot > goalRotations:
        param_coastTimeS -= positionIterationStepSizeS
        debugPrint("decreasing coast")
    else:
        param_coastTimeS += positionIterationStepSizeS
        debugPrint("increasing coast")

degPerStep = 1.8
stepClockHertz = 5 * 1000

stepDiv = positionSeries[-1] / (goalRotations * (360 / degPerStep))
discretizationTimeInterval = int((1 / simTimestepPeriodSeconds) / stepClockHertz)
# unsafe cast to int here, literally couldn't care less because if someone is putting
# weird values in for the sim period or step clock they deserve to read this comment

stepPositions: List[float] = []
stepTimes: List[float] = []

actualStepsCounter = 0

for i in range(0, len(positionSeries)):  # pylint: disable=C0200
    if i % discretizationTimeInterval == 0:
        if positionSeries[i] >= stepDiv * actualStepsCounter:
            if len(stepPositions) != 0:
                stepTimes.append(timeSeries[i])
                stepPositions.append(stepPositions[-1])

            stepTimes.append(timeSeries[i])
            stepPositions.append(actualStepsCounter * stepDiv)

            actualStepsCounter += 1

if actualStepsCounter < (360 * goalRotations) / degPerStep:
    debugPrint("insufficient step clock frequency!")

stepTimes.append(timeSeries[-1])
stepPositions.append(stepPositions[-1])
stepTimes.append(timeSeries[-1])
stepPositions.append(positionSeries[-1])
# don't need to increment step counter because it mistakenly
# counts a step at the beginning, and these cancel out

debugPrint(str(actualStepsCounter) + " steps")

fig, ax = plt.subplots()
fig.subplots_adjust(right=0.75)

twin1 = ax.twinx()
twin2 = ax.twinx()
twin2.spines.right.set_position(("axes", 1.2))

p1 = ax.plot(timeSeries, positionSeries, c="red")
ax.plot(stepTimes, stepPositions, c="purple")
p2 = twin1.plot(timeSeries, velocitySeries, c="green")
p3 = twin2.plot(timeSeries, accelSeries, c="blue")

ax.set(xlabel="time (ms)", ylabel="position (revolutions)")
twin1.set(ylabel="velocity (rev/second)")
twin2.set(ylabel="acceleration (rev/second/second)")

ax.tick_params(axis="y", colors="red")
twin1.tick_params(axis="y", colors="green")
twin2.tick_params(axis="y", colors="blue")

fig.legend(
    handles=[
        mplpatches.Patch(color="red", label="position"),
        mplpatches.Patch(color="blue", label="acceleration"),
        mplpatches.Patch(color="green", label="velocity"),
    ]
)

plt.show()
