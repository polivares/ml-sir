__author__ = "Patricio Andrés Olivares Roncagliolo"
__email__ = "patricio.olivaresr@usm.cl"

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as spi
import peakutils as pk


class SIIR:
    """SIIR Class, based on paper 
    "Dynamical impacts of the coupling in a model of interactive infectious diseases"
    https://doi.org/10.1063/5.0009452

    Attributes:
        SIIR0 (list): Initial conditions for the SIR model.
        params (list): Parameters for the SIR model. It includes the infection rate and recovery rate.
        t_sim (list): List that includes the time range of the simulation.
        SIIR_res (list): Saves the result of the running SIR model (time series) for the given conditions for all the states.
        self.inf1 (list):
        self.inf2 (list):
        peakpos1 (list): Include the list position of all peaks of the time series result.
        peakpos2 (list): Include the list position of all peaks of the time series result.
    """

    def __init__(self, SIIR0, params, t_sim):
        self.SIIR0 = SIIR0  # Initial conditions
        self.params = params  # SIIR configuration parameters: beta1, beta2, delta1, delta2, beta1prime, beta2prime, delta1prime, delta2prime
        self.t_sim = t_sim  # Time of simulation
        self.SIIR_res = None  # Results of Simulation
        self.inf1 = None  # Time series result of disease one
        self.inf2 = None  # Time series result of disease two
        self.peakpos1 = None
        self.peakpos2 = None

    # SIIR equations
    def __SIIR_eqs(self, SIIR0, t, params):
        SSi = SIIR0[0]
        ISi = SIIR0[1]
        SIi = SIIR0[2]
        IIi = SIIR0[3]
        RSi = SIIR0[4]
        SRi = SIIR0[5]
        RIi = SIIR0[6]
        IRi = SIIR0[7]
        RRi = SIIR0[8]

        N = np.sum(SIIR0)

        beta1, beta2, gamma1, gamma2, beta1prime, beta2prime, gamma1prime, gamma2prime = params

        SS = -SSi * beta1 * (ISi + IIi + IRi) / N - SSi * \
            beta2 * (SIi + IIi + RIi) / N
        IS = SSi * beta1 * (ISi + IIi + IRi) / N - gamma1 * \
            ISi - ISi * beta2prime * (SIi + IIi + RIi) / N
        SI = SSi * beta2 * (SIi + IIi + RIi) / N - gamma2 * \
            SIi - SIi * beta1prime * (ISi + IIi + IRi) / N
        II = ISi * beta2prime * (SIi + IIi + RIi) / N + SIi * beta1prime * (
            ISi + IIi + IRi) / N - gamma1prime * IIi - gamma2prime * IIi
        RS = gamma1 * ISi - RSi * beta2 * (SIi + IIi + RIi) / N
        SR = gamma2 * SIi - SRi * beta1 * (ISi + IIi + IRi) / N
        RI = RSi * beta2 * (SIi + IIi + RIi) / N + \
            gamma1prime * IIi - gamma2 * RIi
        IR = SRi * beta1 * (ISi + IIi + IRi) / N + \
            gamma2prime * IIi - gamma1 * IRi
        RR = gamma1 * IRi + gamma2 * RIi

        return SS, IS, SI, II, RS, SR, RI, IR, RR

    # Configure SIIR model with initial conditions, parameters and time
    def __modelSIIR(self, SIIR0, t, params):
        SIIR_res = spi.odeint(self.__SIIR_eqs, SIIR0, t, args=(params,))
        return SIIR_res

    # Run simulation
    def runEvaluation(self, norm=False):
        self.SIIR_res = self.__modelSIIR(self.SIIR0, self.t_sim, self.params)
        N = np.sum(self.SIIR0)
        if norm:
            self.SIIR_res = self.SIIR_res/N
        self.inf1 = self.SIIR_res[:, 1] + \
            self.SIIR_res[:, 3] + self.SIIR_res[:, 7]
        self.inf2 = self.SIIR_res[:, 2] + \
            self.SIIR_res[:, 3] + self.SIIR_res[:, 6]
        self.peakpos1 = pk.indexes(self.dis1, thres=0.5)
        self.peakpos2 = pk.indexes(self.dis2, thres=0.5)

    # Get Time Series of both diseases (whole result)
    def getResult(self):
        return self.SIIR_res

    # Get Time Series of Disease one
    def getDisease1(self):
        return self.inf1, self.peakpos1

    # Get Time Series of Disease two
    def getDisease2(self):
        return self.inf2, self.peakpos2

    def getNInfected1(self):
        # Number of infected of disease 1. All the recovered plus all the active infected at
        # end time
        n_infected = self.SIIR_res[:, 4][-1] + \
                self.SIIR_res[:, 6][-1] + self.SIIR_res[:, 8][-1] + \
                self.SIIR_res[:, 1][-1] + self.SIIR_res[:, 3][-1] + \
                self.SIIR_res[:, 7][-1]
                
        return n_infected

    def getNInfected2(self):
        # Number of infected of disease 2. All the recovered plus all the active infected at
        # end time
        n_infected = self.SIIR_res[:, 5][-1] + \
                self.SIIR_res[:, 7][-1] + self.SIIR_res[:, 8][-1] + \
                self.SIIR_res[:, 2][-1] + self.SIIR_res[:, 3][-1] + \
                self.SIIR_res[:, 6]
        return n_infected
