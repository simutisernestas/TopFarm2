from abc import ABC, abstractmethod
from openmdao.core.explicitcomponent import ExplicitComponent


class Constraint(ABC):

    @abstractmethod
    def setup_as_constraint(self, problem, **kwargs):  # pragma: no cover
        pass

    @abstractmethod
    def setup_as_penalty(self, problem, **kwargs):  # pragma: no cover
        pass

    # It's common for _setup methods within concrete constraints to be called by
    # setup_as_constraint/penalty. Those _setup methods will need to be adapted
    # to receive and use turbine_diameter from kwargs.

    @property
    def constraintComponent(self):
        return self.comp


class ConstraintComponent(ExplicitComponent, ABC):
    def __init__(self, **kwargs):
        ExplicitComponent.__init__(self, **kwargs)

    @abstractmethod
    def satisfy(self, state):  # pragma: no cover
        pass

    def plot(self, ax):
        pass
