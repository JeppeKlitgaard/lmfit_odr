from lmfit import Model as LmfitModel
from lmfit.model import ModelResult as LmfitModelResult
from lmfit import Parameters as LmfitParameters
from lmfit.minimizer import eval_stderr

from uncertainties import ufloat, UFloat

import numpy.typing as npt
import numpy as np

from typing import Callable, Any, Literal

from scipy.odr import Model as ODRModel
from scipy.odr import RealData as ODRRealData
from scipy.odr import ODR


def fit(
    lmfit_model: LmfitModel,
    data: npt.ArrayLike,
    lmfit_params: LmfitParameters | None = None,
    weights_x: npt.ArrayLike | None = None,
    weights_y: npt.ArrayLike | None = None,
    method: str | None = None,
    iter_cb: Callable[..., Any] | None = None,
    scale_covar: bool | None = None,
    verbose: bool | None = None,
    fit_kws: dict[str, Any] | None = None,
    nan_policy: Literal["raise", "propagate", "omit"] | None = None,
    calc_covar: bool | None = None,
    max_nfev: int | None = None,
    **kwargs: Any,
) -> LmfitModelResult:
    # Check that invalid parameters are not supplied
    _invalid_param_err = (
        "{name} is reserved to reflect the lmfit API, but cannot be used."
    )
    if method is not None:
        raise NotImplementedError(_invalid_param_err.format(name="method"))

    if iter_cb is not None:
        raise NotImplementedError(_invalid_param_err.format(name="iter_cb"))

    if scale_covar is not None:
        raise NotImplementedError(_invalid_param_err.format(name="scale_covar"))

    if verbose is not None:
        raise NotImplementedError(_invalid_param_err.format(name="verbose"))

    if nan_policy is not None:
        raise NotImplementedError(_invalid_param_err.format(name="nan_policy"))

    if calc_covar is not None:
        raise NotImplementedError(_invalid_param_err.format(name="calc_covar"))

    if lmfit_params is None:
        lmfit_params = lmfit_model.make_params()

    fit_kws = fit_kws or {}

    # Prep uncertainties
    odr_sx = None if weights_x is None else np.reciprocal(np.sqrt(weights_x))
    odr_sy = None if weights_y is None else np.reciprocal(np.sqrt(weights_y))

    # Prep independent variables
    if not len(lmfit_model.independent_vars) == 1:
        raise ValueError(
            "lmfit_odr only supports a single independent variable at this time."
        )

    indep_var_name = lmfit_model.independent_vars[0]
    if not indep_var_name in kwargs:
        raise ValueError(
            f"Must provide data for independent variable {indep_var_name}."
        )

    # We need to make a mapping between beta array and lmfit parameters
    variable_param_names = [name for name, param in lmfit_params.items() if param.vary]
    nfev = 0

    def odr_func(beta: npt.ArrayLike, x: npt.ArrayLike) -> npt.ArrayLike:
        # Check if we need to stop
        nonlocal nfev
        if nfev == max_nfev:
            raise StopIteration

        wrapped_params = dict(zip(variable_param_names, beta))
        wrapped_params[indep_var_name] = x

        output = lmfit_model.eval(**wrapped_params)
        nfev += 1

        return output

    # First we must construct an ODRModel using information in our LmfitModel
    odr_model = ODRModel(odr_func)

    # Next we must construct a data object
    odr_data = ODRRealData(x=kwargs[indep_var_name], y=data, sx=odr_sx, sy=odr_sy)

    # Prep initial parameter guesses
    odr_beta0 = [lmfit_params[param_name].value for param_name in variable_param_names]
    odr_obj = ODR(data=odr_data, model=odr_model, beta0=odr_beta0, **fit_kws)

    odr_output = odr_obj.run()

    # Construct lmfit ModelResult
    def _error_raiser(*args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(
            "This function is not defined for lmfit_odr ModelResults."
        )

    lmfit_result_params = lmfit_params.copy()
    for i, param_name in enumerate(variable_param_names):
        lmfit_result_params[param_name].value = odr_output.beta[i]
        lmfit_result_params[param_name].stderr = odr_output.sd_beta[i]
        lmfit_result_params[param_name].odr_cov_beta = odr_output.cov_beta[i]

    # Evaluate uncertainties on expr variables
    uvars: list[UFloat] = []
    uvar_names: list[str] = []
    for param_name, param in lmfit_result_params.items():
        if param.stderr is None:
            continue

        uvar = ufloat(param.value, param.stderr)
        uvars.append(uvar)
        uvar_names.append(param.name)

    for param_name, param in lmfit_result_params.items():
        if not param.vary and param.expr:
            eval_stderr(
                lmfit_result_params[param_name],
                uvars,
                uvar_names,
                lmfit_result_params,
            )

    lmfit_model_result = LmfitModelResult(
        model=lmfit_model,
        params=lmfit_result_params,
        data=data,
        weights=weights_y,  # Note: We lose information here
        method=method,
        fcn_args=None,
        fcn_kws=kwargs,
        iter_cb=iter_cb,
        scale_covar=scale_covar,
        nan_policy=nan_policy,
        calc_covar=calc_covar,
        max_nfev=max_nfev,
        **fit_kws,
    )

    lmfit_model_result.fit = _error_raiser

    # Finalise fit and give lmfit_model_result needed attributes
    lmfit_model_result.nfev = nfev
    lmfit_model_result.nvarys = len(variable_param_names)
    lmfit_model_result.nfree = len(odr_output.y) - lmfit_model_result.nvarys
    lmfit_model_result.redchi = odr_output.res_var or 0.0
    lmfit_model_result.chisqr = lmfit_model_result.redchi * lmfit_model_result.nfree

    # Todo: acutally calculate these
    lmfit_model_result.aic = 0.0
    lmfit_model_result.bic = 0.0

    return lmfit_model_result
