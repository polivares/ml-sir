__author__ = "Patricio Andrés Olivares Roncagliolo"
__email__ = "patricio.olivaresr@usm.cl"

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as spi
import peakutils as pk
# Exceptions from model
from models.exceptions import NotEvaluatedError


class SIR:
    """SIR Class that run the original SIR model

    Attributes:
        SIR0 (list): Initial conditions for the SIR model.
        params (list): Parameters for the SIR model. It includes the infection rate and recovery rate.
        t_sim (list): List that includes the time range of the simulation.
        SIR_res (list): Saves the result of the running SIR model (time series) for the given conditions for all the states.
        peakpos (list): Include the list position of all peaks of the time series result.
    """

    def __init__(self, SIR0, params, t_sim) -> None:
        """Constructor

        :param SIR0: Initial conditions for the SIR model.
        :type SIR0: list
        :param params: Parameters for the SIR model. It includes the infection rate and recovery rate.
        :type params: list
        :param t_sim: List that includes the time range of the simulation.
        :type t_sim: list
        """
        self.SIR0 = SIR0
        self.params = params
        self.t_sim = t_sim
        self.SIR_res = None
        self.peakpos = None
        self.__evaluated = False  # Internal attribute for checking evaluation made

    def __SIR_eqs(self, SIR0, t, params):
        """Private method that include the SIR model equations

        :param SIR0: Initial conditions for the SIR model.
        :type SIR0: list
        :param t: List that includes the time range of the simulation.
        :type t: list
        :param params: Parameters for the SIR model. It includes the infection rate and recovery rate.
        :type params: list
        :return: The result for the Susceptible (S), Infected (I) and Recovery (R) states
        :rtype: list
        """

        # Initial conditions
        Si = SIR0[0]
        Ii = SIR0[1]
        Ri = SIR0[2]

        # Number of infected people
        N = np.sum(SIR0)

        # Parameters of SIR model
        beta, gamma = params

        # Equations definition of the model
        S = - (beta * Si * Ii) / N
        I = (beta * Si * Ii) / N - gamma * Ii
        R = gamma * Ii

        return S, I, R

    def __model_evaluated(self):
        """Private method for running verification before obtaining statistics from model.

        :raises NotEvaluatedError: Exception raise when model has not be executed yet.
        """
        if not self.__evaluated:
            raise NotEvaluatedError(
                "Evaluation of model must be executed before")

    # Configure SIR model with initial conditions, parameters and time
    def __modelSIR(self, SIR0, t, params):
        """Private method that configure the SIR model with initial conditions, parameters and time.

        :param SIR0: Initial conditions for the SIR model.
        :type SIR0: list
        :param t: List that includes the time range of the simulation.
        :type t: list
        :param params: Parameters for the SIR model. It includes the infection rate and recovery rate.
        :type params: list
        :return: Final result of the evaluation using the SIR equations (SIR_res)
        :rtype: 2D-list
        """
        SIR_res = spi.odeint(self.__SIR_eqs, SIR0, t, args=(params,))
        self.__evaluated = True
        return SIR_res

    def run(self, norm=False):
        """Run the evaluation of the model and saves the result on SIR_Res variable and the peak position of the infection
        of the disease on peakpos variable.

        :param norm: If True, simulation is executed with normalized values. Defaults to False
        :type norm: bool, optional
        """
        self.SIR_res = self.__modelSIR(self.SIR0, self.t_sim, self.params)
        self.__model_evaluated()
        if norm:
            N = np.sum(self.SIR0)
            self.SIR_res = self.SIR_res / N
        else:
            self.SIR_res = np.rint(self.SIR_res).astype(int)
        
        # Check for peak infection position and save it
        self.peakpos = pk.indexes(self.SIR_res[:, 1], thres=0.5)

    def getResult(self):
        """Return the result of the evaluation (SIR_res), corresponding to the states 
        Susceptible, Infection y Recovered.

        Data for each state:
        S = SIR_res[:, 0] 
        I = SIR_res[:, 1]
        R = SIR_res[:, 2]  

        :return: Final result of the evaluation using the SIR equations (SIR_res)
        :rtype: 2D-list
        """
        self.__model_evaluated()
        return self.SIR_res

    def getPeakPos(self):
        """Return the peak indexes position of the infection state (I) from disease behavior. This position
        can be translated to time using t_sim array or to a value using SIR_res[:, 1]

        :return: Peak index position of disease
        :rtype: list
        """
        self.__model_evaluated()
        return self.peakpos

    def getDisease(self):
        """Return a list only including the progress of the infected state.

        :return: List with progress of infected state.
        :rtype: list
        """
        self.__model_evaluated()
        return self.SIR_res[:, 1]

    def getNInfected(self):
        """Number of people infected at the end of the simulation

        :return: Number of people infected
        :rtype: float
        """
        self.__model_evaluated()
        # All the recovered plus the still infected people at the last simulation time
        n_infected = self.SIR_res[:, 2][-1] + self.SIR_res[:, 1][-1]
        return n_infected
