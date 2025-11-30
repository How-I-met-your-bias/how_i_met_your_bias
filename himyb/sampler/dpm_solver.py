"""
Nathan Roos

"""

import torch
import numpy as np


def dpm_solver_1(
    model: torch.nn.Module,
    n_steps: int,
    class_labels: torch.Tensor,
    batch_shape,
    sigma_min: float,
    sigma_max: float,
    device,
    x_init=None,
    return_history: bool = False,
    order=1,
    eps_s=1e-3,
    beta_d=0.1,
    beta_0=19.9,
    guidance_scale: float = 0,
    use_unconditional_model=False,
):
    """
    Implementation of DPM-solver-1 for diffusion models trained with the Karras framework.
    Uses the VP assumption of DPM, i.e., alpha_dpm(t)^2 + sigma_dpm(t)^2 = 1.
    The noise schedule is a uniform repartition of the lambda_dpm.

    Args
        model (torch.nn.Module): The trained diffusion model. Should have signature model(x, sigma, class_labels) -> x_denoised.
        n_steps (int): Number of steps to use for the sampling process.
        class_labels (torch.Tensor): Class labels for class-conditional models, shape (batch_size,).
        batch_shape (tuple): Shape of the batch to generate (e.g., (batch_size, n_channels, height, width)).
        sigma_min (float): Minimum noise level for the noise schedule.
        sigma_max (float): Maximum noise level for the noise schedule.
        device (torch.device): Device to use for computations.
        x_init (torch.Tensor, optional): Initial noise tensor. If None, a random noise tensor is generated. Defaults to None.
        return_history (bool, optional): Whether to return the history of generated images at each step. Defaults to False.
        order (int, optional): Order of the solver. Only order=1 is supported. Defaults to 1.
        eps_s (float, optional): Small epsilon parameter for schedule calculation. Defaults to 1e-3.
        beta_d (float, optional): Beta_d parameter for schedule calculation. Defaults to 0.1.
        beta_0 (float, optional): Beta_0 parameter for schedule calculation. Defaults to 19.9.
        guidance_scale (float, optional): Guidance scale for classifier-free guidance. Defaults to 0.
    Returns:
        dict: Dictionary containing:
            - "generated_imgs": Final generated images (torch.Tensor).
            - "sigma_schedule": Noise schedule used (torch.Tensor).
            - "lambda_schedule": Lambda schedule used (torch.Tensor).
            - "n_steps": Number of steps performed (int).
            - "history" (optional): Dictionary mapping sigma values to intermediate images (if return_history=True).
    Note:
        This implementation assumes the model follows the EDM/DPM conventions and is compatible with the Karras noise schedule.
    """
    assert order in [1]
    model.to(device).eval()
    batch_size = batch_shape[0]
    class_labels = class_labels.to(device)
    assert class_labels.shape[0] == batch_size
    if x_init is not None:
        assert tuple(x_init.shape) == tuple(batch_shape)

    def noise_pred_model(x, sigma, class_lab):
        """Convert model output (denoised image) to noise prediction"""
        model_output = model(x, torch.tile(sigma, dims=(batch_size, 1)), class_lab)
        return (x - model_output) / sigma

    def dpm_alpha_from_edm_sigma(sigma):
        """Compute alpha (scaling) in DPM from sigma (noise level) in EDM"""
        return 1.0 / torch.sqrt(1.0 + torch.pow(sigma, 2))

    def dpm_lambda_from_edm_sigma(sigma):
        """Compute lambda(t) in DPM from sigma(t) in EDM"""
        return -torch.log(sigma)

    def edm_sigma_from_dpm_lambda(lmbda):
        """Compute sigma(t) in EDM from lambda(t) in DPM"""
        return torch.exp(-lmbda)

    def dpm_sigma_from_edm_sigma(sigma):
        """Compute sigma_dpm(t) in DPM from sigma(t) in EDM"""
        return sigma / torch.sqrt(1.0 + torch.pow(sigma, 2))

    betad_from_sigma = (
        lambda sig_min, sig_max, eps_s: 2
        * (
            torch.log(torch.tensor(sig_min) ** 2 + 1) / eps_s
            - torch.log(torch.tensor(sig_max) ** 2 + 1)
        )
        / (eps_s - 1)
    )
    beta0_from_sigma = (
        lambda sig_max, beta_d: torch.log(torch.tensor(sig_max) ** 2 + 1) - beta_d * 0.5
    )
    t_from_lambda = (
        lambda beta_d, beta_0, lmbda: 2
        * torch.log(torch.exp(-2 * lmbda) + 1)
        / (
            torch.sqrt(beta_0**2 + 2 * beta_d * torch.log(torch.exp(-2 * lmbda) + 1))
            + beta0
        )
    )

    # initial and final noise levels
    betad = torch.tensor(beta_d)  # betad_from_sigma(sigma_min, sigma_max, eps_s)
    beta0 = torch.tensor(beta_0)  # beta0_from_sigma(sigma_max, betad)
    sigma_min = max(
        sigma_min, (np.e ** (0.5 * betad * (eps_s**2) + beta0 * eps_s) - 1) ** 0.5
    )
    sigma_max = min(sigma_max, (np.e ** (0.5 * betad + beta0) - 1) ** 0.5)
    sigma_min = torch.tensor(sigma_min, device=device)
    sigma_max = torch.tensor(sigma_max, device=device)

    lambda_min = dpm_lambda_from_edm_sigma(sigma_max)
    lambda_max = dpm_lambda_from_edm_sigma(sigma_min)
    nfe = 0

    n_classes = model.label_dim - 1

    # compute the noise levels schedule in terms of DPM lambda
    lambda_schedule = torch.linspace(lambda_min, lambda_max, n_steps + 1, device=device)
    # convert DPM lambda schedule to EDM sigma schedule
    sigma_schedule = edm_sigma_from_dpm_lambda(lambda_schedule)
    # initialize the latent variables (initial noise)
    if x_init is None:
        x_i = torch.randn(batch_shape, device=device) * dpm_sigma_from_edm_sigma(
            sigma_schedule[0]
        )
    else:
        x_i = x_init.to(device)

    history = {}
    if return_history:
        history[sigma_schedule[0].item()] = x_i.detach().cpu()
    if use_unconditional_model:
        uncond_class_labels = torch.zeros_like(class_labels)

    with torch.inference_mode():
        # loop over the n_steps noise levels (sigma schedule)
        for i_step in range(n_steps):
            # compute alpha_i, alpha_{i+1}, lambda_i, lambda_{i+1}
            sigma_i, sigma_ip1 = sigma_schedule[i_step], sigma_schedule[i_step + 1]
            dpm_sigma_ip1 = dpm_sigma_from_edm_sigma(sigma_ip1)
            dpm_sigma_i = dpm_sigma_from_edm_sigma(sigma_i)
            alpha_i = dpm_alpha_from_edm_sigma(sigma_i)
            alpha_ip1 = dpm_alpha_from_edm_sigma(sigma_ip1)
            lambda_i, lambda_ip1 = lambda_schedule[i_step], lambda_schedule[i_step + 1]
            h_ip1 = lambda_ip1 - lambda_i

            # compute the noise prediction at step i
            pred_noise = noise_pred_model(x_i, dpm_sigma_i, class_labels)
            if guidance_scale > 0:
                if use_unconditional_model:
                    uncond_pred_noise = noise_pred_model(
                        x_i, dpm_sigma_i, uncond_class_labels
                    )
                    nfe += 1
                else:  # aggregate the score of other classes
                    # we aggregate the scores of all the classes
                    uncond_pred_noise = pred_noise.clone() / n_classes
                    # we loop over all the other classes as we need to compute the average of their scores
                    for j in range(1, n_classes):
                        # we shift the class labels to go to the next class, avoiding 0
                        tmp_class_labels = (
                            ((class_labels.clone() - 1) + j) % n_classes
                        ) + 1
                        uncond_pred_noise += (
                            noise_pred_model(x_i, dpm_sigma_i, tmp_class_labels)
                            / n_classes
                        )
                        nfe += 1
                pred_noise = (
                    1 + guidance_scale
                ) * pred_noise - guidance_scale * uncond_pred_noise

            nfe += 1

            # compute the next denoised image x_i
            # print(f"Step {i_step+1} : sigma_i={sigma_i.item():.4f}, alpha_i={alpha_i.item():.4f}, lambda_i={lambda_i.item():.4f}")
            if order == 1:
                x_ip1 = (alpha_ip1 / alpha_i) * x_i - (
                    dpm_sigma_ip1 * (torch.exp(h_ip1) - 1.0)
                ) * pred_noise
            elif (
                order == 2
            ):  # not working, presumably because of the mapping between t and edm_sigma(t)
                s_ip1 = t_from_lambda(betad, beta0, (lambda_ip1 + lambda_i) * 0.5)
                sigma_s_ip1 = edm_sigma_from_dpm_lambda((lambda_ip1 + lambda_i) * 0.5)
                u_i = (
                    dpm_alpha_from_edm_sigma(sigma_s_ip1) / alpha_i * x_i
                    - sigma_s_ip1 * (torch.exp(h_ip1 / 2.0) - 1.0) * pred_noise
                )
                pred_noise_s = noise_pred_model(
                    u_i, dpm_sigma_from_edm_sigma(sigma_s_ip1)
                )
                nfe += 1
                x_ip1 = (alpha_ip1 / alpha_i) * x_i - dpm_sigma_ip1 * (
                    torch.exp(h_ip1) - 1.0
                ) * pred_noise_s
            else:
                raise NotImplementedError(
                    f"DPM-Solver of order {order} not implemented"
                )

            if return_history:
                history[sigma_ip1.item()] = x_ip1.cpu()
            x_i = x_ip1
        result = {
            "generated_imgs": x_ip1.cpu(),
            "sigma_schedule": sigma_schedule,
            "lambda_schedule": lambda_schedule,
            "n_steps": n_steps,
        }
        if return_history:
            result["history"] = history
        return result
