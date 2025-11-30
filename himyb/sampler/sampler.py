"""
Nathan Roos

Implementation of the deterministic and stochastic version of the sampling algorithm proposed
in Karras 2022. It is inspired by the original implementation in https://github.com/NVlabs/edm.

As the deterministic version is a special case of the stochastic version (with S_churn = 0),
we advise always using the stochastic version for clarity and to avoid errors, even when S_churn = 0.
"""

import torch
import numpy as np


def stoch_edm_sampler(
    model,
    class_labels,
    num_steps,
    shape,
    device,
    rho=7,
    return_history=False,
    x_init=None,
    sigma_min=0.002,
    sigma_max=80,
    s_churn=40,
    s_min=0.05,
    s_max=50,
    s_noise=1.003,
    use_heun=True,
    guidance_scale=0.0,
    use_unconditional_model=False,
    n_skipped_steps=0
):
    """
    Stochastic sampler (algorithm 2) from Karras et al. 2022.

    Args:
        model (torch.nn.Module): model to sample from
        class_labels (torch.Tensor): Shape (N, ) class labels for the model
        num_steps (int): Number of integration steps
        shape : Shape of the batch of images to generate (ie : (N, C, H, W)),\
              must match the model specifications
        device (torch.device): Device on which to run the integration
        rho (int): Exponent for the time discretization. Higher values will\
                result in more steps at the end of the integration. Karras 2022\
                uses rho = 7.
        return_history (bool): If True, the function will also return the history of\
                the generated images at each step of the integration.
        x_init (torch.Tensor): Initial noise image to start the integration from. If None,\
                a random image will be generated. If provided, it must have the same shape\
                as the generated images.
        sigma_min (float): Minimum value of the noise schedule (default value corresponds to the implemented models)
        sigma_max (float): Maximum value of the noise schedule (default value corresponds to the implemented models)
        s_churn (float): Control the variance of the noise added to the image at each step.\
        s_min (float): Minimum time step such that we use stochasticity
        s_max (float): Maximum time step such that we use stochasticity
        s_noise (float): inflation parameter of the newly added noise. The default value comes from\
                the Karras 2022 paper and has not been grid searched.
        use_heun (bool): If True, the integration will be done using Heun's method.\
                If False, the integration will be done using Euler's method.
        guidance_scale (float): Scale for the classifier-free guidance. If 0, no guidance is applied.
        use_unconditional_model (bool): If True, when using CFG, the unconditional score is computed\
            using the model conditioned on class_labels == 0. If False, the unconditional score is\
            computed using the average of the scores of all the conditioning classes (it is 
            slower because we call the network more times at each step). Only set to False if the\
            unconditional class is not available or not good.
        n_skipped_steps (int): Number of steps to skip at the beginning of the denoising.\
    """
    use_cfg = guidance_scale > 0
    n_classes = model.label_dim - 1

    model.eval().to(device)
    with torch.inference_mode():
        batch_size = shape[0]
        step_indices = torch.arange(num_steps, device=device)
        # construct time steps [|0, N-1|]
        t_steps = (
            sigma_max ** (1 / rho)
            + step_indices
            / (num_steps - 1)
            * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
        ) ** rho
        # add t_N = 0
        t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])
        if n_skipped_steps > 0:
            # skip the first n_skipped_steps steps
            t_steps = torch.cat((t_steps[:1], t_steps[n_skipped_steps + 1:]))
            num_steps -= n_skipped_steps

        ## main sampling loop
        # cur refers to index i and next refers to index i+1
        if x_init is not None:
            x_next = x_init.clone()
        else:
            x_next = torch.randn(shape, device=device) * sigma_max
        x_cur = x_next
        nfe = 0  # number of function evaluations
        if return_history:
            history = {t_steps[0].item(): x_cur.clone().detach()}

        class_labels = class_labels.to(device)
        if use_cfg:
            mask_cond_imgs = class_labels != 0  # all the images that have a class label
            if use_unconditional_model:
                uncond_class_labels = torch.zeros_like(class_labels[mask_cond_imgs])
            num_cond_imgs = int(mask_cond_imgs.sum())
        for i in range(0, num_steps):
            x_cur = x_next
            t_cur = t_steps[i]
            t_next = t_steps[i + 1]

            # increase noise temporarily
            if s_min <= t_cur <= s_max:
                #gamma is clamped so that we never add more noise than there currently is
                gamma = min(s_churn / num_steps, np.sqrt(2) - 1)
                t_hat = torch.as_tensor(t_cur * (1 + gamma))
                x_hat = x_cur + torch.sqrt(
                    t_hat**2 - t_cur**2
                ) * s_noise * torch.randn_like(x_cur)
            else:
                gamma = 0
                t_hat = t_cur
                x_hat = x_cur

            denoised_imgs = model(
                x_hat, torch.tile(t_hat, dims=(batch_size, 1)), class_labels
            )
            nfe += 1
            d_cur = (x_hat - denoised_imgs) / t_hat

            if use_cfg:  # when using classifier-free guidance
                if use_unconditional_model:
                    uncond_denoised_imgs = model(
                        x_hat[mask_cond_imgs],
                        torch.tile(t_hat, dims=(num_cond_imgs, 1)),
                        uncond_class_labels,
                    )
                    nfe += 1
                    # only apply guidance to the images that have a class label
                    d_cur[mask_cond_imgs] = (1 + guidance_scale) * d_cur[
                        mask_cond_imgs
                    ] - guidance_scale * (
                        x_hat[mask_cond_imgs] - uncond_denoised_imgs
                    ) / t_hat
                else:
                    # we aggregate the scores of all the classes
                    avg_d_cur = d_cur[mask_cond_imgs].clone() / n_classes
                    # we loop over all the other classes as we need to compute the average of their scores
                    for j in range(1, n_classes):
                        # we shift the class labels to go to the next class, avoiding 0
                        tmp_class_labels = (
                            ((class_labels[mask_cond_imgs].clone() - 1) + j) % n_classes
                        ) + 1
                        tmp_denoised_imgs = model(
                            x_hat[mask_cond_imgs],
                            torch.tile(t_hat, dims=(num_cond_imgs, 1)),
                            tmp_class_labels,
                        )
                        nfe += 1
                        avg_d_cur += (x_hat[mask_cond_imgs] - tmp_denoised_imgs) / (
                            t_hat * n_classes
                        )
                    d_cur[mask_cond_imgs] = (1 + guidance_scale) * d_cur[
                        mask_cond_imgs
                    ] - guidance_scale * avg_d_cur

            x_next = x_hat + (t_next - t_hat) * d_cur
            if use_heun and i < num_steps - 1:
                denoised_imgs = model(
                    x_next, torch.tile(t_next, dims=(batch_size, 1)), class_labels
                )
                nfe += 1
                d_cur_prime = (x_next - denoised_imgs) / t_next

                if use_cfg:
                    if use_unconditional_model:
                        uncond_denoised_imgs = model(
                            x_next[mask_cond_imgs],
                            torch.tile(t_next, dims=(num_cond_imgs, 1)),
                            uncond_class_labels,
                        )
                        nfe += 1
                        # only apply guidance to the images that have a class label
                        d_cur_prime[mask_cond_imgs] = (
                            1 + guidance_scale
                        ) * d_cur_prime[mask_cond_imgs] - guidance_scale * (
                            x_next[mask_cond_imgs] - uncond_denoised_imgs
                        ) / t_next
                    else:
                        # we aggregate the scores of all the classes
                        avg_d_cur_prime = (
                            d_cur_prime[mask_cond_imgs].clone() / n_classes
                        )
                        # we loop over all the other classes as we need to compute the average of their scores
                        for j in range(1, n_classes):
                            # we shift the class labels to go to the next class, avoiding 0
                            tmp_class_labels = (
                                ((class_labels[mask_cond_imgs].clone() - 1) + j)
                                % n_classes
                            ) + 1
                            tmp_denoised_imgs = model(
                                x_next[mask_cond_imgs],
                                torch.tile(t_next, dims=(num_cond_imgs, 1)),
                                tmp_class_labels,
                            )
                            nfe += 1
                            avg_d_cur_prime += (
                                x_next[mask_cond_imgs] - tmp_denoised_imgs
                            ) / (t_next * n_classes)
                        d_cur_prime[mask_cond_imgs] = (
                            1 + guidance_scale
                        ) * d_cur_prime[
                            mask_cond_imgs
                        ] - guidance_scale * avg_d_cur_prime

                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_cur_prime)
            if return_history:
                history[t_cur.item()] = x_cur.clone().detach()
        result = {"generated_imgs": x_next, "NFE": nfe, "t_steps": t_steps}
        if return_history:
            result["history"] = history
    return result



def det_edm_sampler(
    model,
    class_labels,
    num_steps,
    shape,
    device,
    rho=7,
    return_history=False,
    x_init=None,
    use_heun=True,
    guidance_scale=0.0,
    use_unconditional_model=False,
):
    """
    Deterministic version of the sampling algorithm proposed in Karras 2022.
    It integrates the ODE using Heun's second order method. The ODE is defined as:
    dx/dt = - (f(x,t) - x) / t
    where f is the model

    Args:
        model (torch.nn.Module): model to sample from
        class_labels (torch.Tensor): Shape () class labels for the model
        num_steps (int): Number of integration steps
        shape : Shape of the batch of images to generate (eg : (10, 3, 32, 32)),\
              must match the model specifications
        device (torch.device): Device on which to run the integration
        rho (int): Exponent for the time discretization. Higher values will\
                result in more steps at the end of the integration.
        return_history (bool): If True, the function will also return the history of\
                the generated images at each step of the integration.
        x_init (torch.Tensor): Initial image to start the integration from. If None,\
                a random image will be generated. If provided, it must have the same shape\
                as the generated images.
        use_heun (bool): If True, the integration will be done using Heun's method.\
                If False, the integration will be done using Euler's method.
        guidance_scale (float): Scale for the classifier-free guidance. If 0, no guidance is applied.
        use_unconditional_model (bool): If True, when using CFG, the unconditional score is computed\
            using the model conditioned on class_labels == 0. If False, the unconditional score is\
            computed using the average of the scores of all the conditioning classes.
    Returns:
        dict: A dictionary containing : a list of the generated images "generated_imgs" and the number of \
            function evaluations "NFE" done during the integration. If return_history is True,\
            the dictionary will also contain the history of the generated images under the key \
                "history".
    """
    use_cfg = guidance_scale > 0

    if use_cfg and (not use_unconditional_model):
        raise NotImplementedError("Pls use the stochastic sampler for this case")

    model.eval().to(device)
    with torch.inference_mode():
        sigma_min = 0.002
        sigma_max = 80
        batch_size = shape[0]
        step_indices = torch.arange(num_steps, device=device)
        # construct time steps [|0, N-1|]
        t_steps = (
            sigma_max ** (1 / rho)
            + step_indices
            / (num_steps - 1)
            * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
        ) ** rho
        # add t_N = 0
        t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])

        ## main sampling loop
        # cur refers to index i and next refers to index i+1
        if x_init is not None:
            x_next = x_init.clone()
        else:
            x_next = torch.randn(shape, device=device) * sigma_max
        x_cur = x_next
        nfe = 0  # number of function evaluations
        if return_history:
            history = [x_cur.clone().detach()]

        class_labels = class_labels.to(device)
        if use_cfg:
            mask_cond_imgs = class_labels != 0  # all the images that have a class label
            uncond_class_labels = torch.zeros_like(class_labels[mask_cond_imgs])
            num_cond_imgs = int(mask_cond_imgs.sum())
        for i in range(0, num_steps):
            x_cur = x_next
            t_cur = t_steps[i]
            t_next = t_steps[i + 1]

            denoised_imgs = model(
                x_cur, torch.tile(t_cur, dims=(batch_size, 1)), class_labels
            )
            nfe += 1
            d_cur = (x_cur - denoised_imgs) / t_cur

            if use_cfg:  # when using classifier-free guidance
                # denoised images for the unconditioned model
                uncond_denoised_imgs = model(
                    x_cur[mask_cond_imgs],
                    torch.tile(t_cur, dims=(num_cond_imgs, 1)),
                    uncond_class_labels,
                )
                nfe += 1
                # only apply guidance to the images that have a class label
                d_cur[mask_cond_imgs] = (1 + guidance_scale) * d_cur[
                    mask_cond_imgs
                ] - guidance_scale * (
                    x_cur[mask_cond_imgs] - uncond_denoised_imgs
                ) / t_cur

            x_next = x_cur + (t_next - t_cur) * d_cur
            if use_heun and i < num_steps - 1:
                denoised_imgs = model(
                    x_next, torch.tile(t_next, dims=(batch_size, 1)), class_labels
                )
                nfe += 1
                d_cur_prime = (x_next - denoised_imgs) / t_next
                if use_cfg:
                    # denoised images for the unconditioned model
                    uncond_denoised_imgs = model(
                        x_next[mask_cond_imgs],
                        torch.tile(t_next, dims=(num_cond_imgs, 1)),
                        uncond_class_labels,
                    )
                    nfe += 1
                    # only apply guidance to the images that have a class label
                    d_cur_prime[mask_cond_imgs] = (1 + guidance_scale) * d_cur_prime[
                        mask_cond_imgs
                    ] - guidance_scale * (
                        x_next[mask_cond_imgs] - uncond_denoised_imgs
                    ) / t_next

                x_next = x_cur + (t_next - t_cur) * (0.5 * d_cur + 0.5 * d_cur_prime)
            if return_history:
                history.append(x_cur.clone().detach())

        result = {"generated_imgs": x_next, "NFE": nfe}
        if return_history:
            result["history"] = history
    return result
