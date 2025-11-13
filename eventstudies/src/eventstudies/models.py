import numpy as np
import statsmodels.api as sm


class Model:

    """
    The Model Class holds various methods to run the various returns models
    It takes the data needed to compute the model as parameters
    and the following parameters:
        est_size: int,
        win_size: int

    Each method returns:
        an array of residuals of the event window
        the degrees of freedom
        an array of the variance of the residuals
        the model used
    """

    def __init__(
        self, est_size: int, win_size: int
    ):
        self.est_size = est_size
        self.win_size = win_size

    def OLS(self, X, Y):
        X = sm.add_constant(X)  # add an intercept

        # trim leading 0 returns from the dataset prior to regression
        l = len(Y)
        Yt = np.trim_zeros(Y)
        lt = len(Yt)
        lz = l - lt
        Y = Yt
        X = X[lz:, :]

        # if there is no values after trimming leading 0s, skip the model
        # are return array of 0 residuals and 0 variance to cope
        if Y[: self.est_size].shape[0] > 0:
            model = sm.OLS(Y[: self.est_size], X[: self.est_size]).fit()
            residuals = np.array(Y) - model.predict(X)
        else:
            model = None
            residuals = np.zeros(np.max([l, self.win_size]))
        var = np.var(residuals[: self.est_size])
        df = self.est_size - 1
        # if residual length is less than win_size after trimming
        # we need to pad the front again so it returns the length 
        # expected
        if len(residuals) < self.win_size:
            pad_size = self.win_size - len(residuals)
            pad = np.zeros(pad_size)
            residuals = np.concatenate([pad, residuals])
        return residuals[-self.win_size:], df, var, model

def MarketModel(
    daily_ret,
    daily_mkt,
    *,  # Named arguments only
    est_size: int,
    win_size: int,
    **kwargs
):
    X = np.array(daily_mkt)
    Y = np.array(daily_ret)
    residuals, df, var_res, model = Model(
        est_size,
        win_size).OLS(
            X,
            Y)
    var = [var_res] * win_size
    return residuals, df, var, model


def FamaFrench3(
    daily_ret,
    Mkt_RF,
    SMB,
    HML,
    RF,
    *,  # Named arguments only
    est_size: int,
    win_size: int,
    **kwargs
):
    RF = np.array(RF)
    Mkt_RF = np.array(Mkt_RF)
    daily_ret = np.array(daily_ret)
    X = np.column_stack((Mkt_RF, SMB, HML))
    Y = np.array(daily_ret) - np.array(RF)
    residuals, df, var_res, model = Model(
        est_size,
        win_size).OLS(
            X, Y)
    var = [var_res] * win_size
    return residuals, df, var, model


def MeanAdjustedModel(
    daily_ret,
    *,  # Named arguments only
    est_size: int,
    win_size: int,
    **kwargs
):
    mean = np.mean(daily_ret[:est_size])
    residuals = np.array(daily_ret) - mean
    df = est_size - 1
    var = [np.var(residuals)] * win_size
    return residuals[-win_size:], df, var, mean


def MarketAdjustedModel(
    daily_ret,
    daily_mkt,
    *,  # Named arguments only
    est_size: int,
    win_size: int,
    **kwargs
):
    X = np.array(daily_mkt)
    Y = np.array(daily_ret)
    residuals = Y - X
    df = est_size - 1
    var = [np.var(residuals)] * win_size
    return residuals[-win_size:], df, var, X[-win_size:]

def OrdinaryReturnsModel(
    daily_ret,
    *,  # Named arguments only
    est_size: int,
    win_size: int,
    **kwargs
):
    X = np.array(daily_ret)
    Y = np.array([0] * len(X))
    residuals = X - Y
    df = est_size - 1
    var = [np.var(residuals)] * win_size
    return residuals[-win_size:], df, var, X[-win_size:]


def FamaFrench5(
    daily_ret,
    Mkt_RF,
    SMB,
    HML,
    RMW,
    CMA,
    RF,
    *,  # Named arguments only
    est_size: int,
    win_size: int,
    **kwargs
):

    RF = np.array(RF)
    Mkt_RF = np.array(Mkt_RF)
    daily_ret = np.array(daily_ret)
    X = np.column_stack((Mkt_RF, SMB, HML, RMW, CMA))
    Y = np.array(daily_ret) - np.array(RF)
    residuals, df, var_res, model = Model(
        est_size,
        win_size).OLS(
            X, Y)
    var = [var_res] * win_size
    return residuals, df, var, model


def Carhart(
    daily_ret,
    Mkt_RF,
    SMB,
    HML,
    RF,
    MOM,
    *,  # Named arguments only
    est_size: int,
    win_size: int,
    **kwargs
):
    RF = np.array(RF)
    Mkt_RF = np.array(Mkt_RF)
    daily_ret = np.array(daily_ret)
    X = np.column_stack((Mkt_RF, SMB, HML, MOM))
    Y = np.array(daily_ret) - np.array(RF)
    residuals, df, var_res, model = Model(
        est_size,
        win_size).OLS(
            X, Y)
    var = [var_res] * win_size
    return residuals, df, var, model 