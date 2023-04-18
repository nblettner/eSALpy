import numpy as np

def gauss2D(y, x, ymu, xmu, sig):
    yp = (1 / (sig * (2 * np.pi) ** (1 / 2))) * np.exp(
        (-1 / 2) * ((y - ymu) / sig) ** 2
    )
    xp = (1 / (sig * (2 * np.pi) ** (1 / 2))) * np.exp(
        (-1 / 2) * ((x - xmu) / sig) ** 2
    )
    return yp * xp


def field_generator(ygrid, xgrid, sigs, ymus, xmus, ampls):
    sigs = np.atleast_1d(sigs)
    ymus = np.atleast_1d(ymus)
    xmus = np.atleast_1d(xmus)
    ampls = np.atleast_1d(ampls)
    
    field = np.zeros(ygrid.shape)
    for i in range(len(ampls)):
        field += ampls[i] * gauss2D(
            ygrid, xgrid, ymus[i], xmus[i], sigs[i]
        )
        
    return field