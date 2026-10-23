"""Copy this file and rename the class to implement another circular-transfer tool.

1. Put the mathematical algorithm in a standalone function with explicit inputs.
2. Implement its applicability checks and explain any rejected request.
3. Convert its result into a ManeuverCandidate with absolute burn epochs and
   delta-v vectors in the request's inertial frame.
4. Add independent mathematical and end-to-end tests.

Do not propagate, plot, change mission state, or choose between tools here.
Use a dedicated request contract when the capability needs different inputs;
LambertRequest is an example. Do not overload a target radius to mean a position
or a target spacecraft.
"""

from .gnc_tool import Applicability
from .models import ManeuverCandidate, TransferRequest


class NewGncTool:
    name = "new_tool"
    description = "Replace with the capability and its assumptions."

    def check_applicability(self, request: TransferRequest) -> Applicability:
        return Applicability(False, "Template only: implement the applicability checks.")

    def propose(self, request: TransferRequest) -> ManeuverCandidate:
        raise NotImplementedError("Copy this template and implement the mathematical method.")
