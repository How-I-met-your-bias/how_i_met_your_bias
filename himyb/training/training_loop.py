"""
Nathan Roos

Implement a fonction that trains a model all the way from scratch
"""

import os
import time
import datetime
import copy

import ml_collections
import torchinfo
import torch

import himyb.misc_utils as misc_utils
import himyb.training.train_misc_utils as train_misc_utils
import himyb.models.ddpmpp as ddpmpp
import himyb.models.preconditioning as preconditioning
import himyb.datasets.get_dataloaders as get_dataloaders
import himyb.training.optim_utils as optim_utils
import himyb.training.loss_utils as loss_utils
import himyb.wandb_utils.wandb_utils as wandb_utils
import himyb.training.save_load as save_load
import himyb.sampler.generate as generate


def training_loop(
    run_dir,
    batch_size: int,
    wandb_wrapper: wandb_utils.WandbWrapper,
    dataset_conf: ml_collections.ConfigDict,
    optimizer_config: ml_collections.ConfigDict,
    model_config: ml_collections.ConfigDict,
    total_kimg: int = 200000,
    report_loss_every_kimg: int = 5,
    save_ckpt_every_kimg: int = 5000,
    save_state_every_min: int = 10,
    use_ema: bool = True,
    ema_halflife_kimg: int = 500,
    ema_rampup_ratio: float = 0.05,
    check_ema_loss_every_kimg: int = 10,
    seed: int = 0,
    device="cuda",
    loss_function: str = "edm",
    resume_training: bool = False,
    states_to_resume: str = None,
    configs_to_resume: str = None,
    use_pretrained_weights: bool = False,
    pretrained_weights_path: str = None,
    use_gradient_clipping: bool = False,
):
    """
    When using resume_training, we resume the training from given states and configurations (keeping
    the last lr, the last nimg, etc.). This is useful to continue training after a crash.
    It is different from using pretrained weights, where we can use the weights of a previous run
    to initialize the model and optimizer, but the training starts from scratch otherwise (n_img is 0,
    lr is the initial lr, etc.).

    Args:
        run_dir (str): directory where the run is saved
        batch_size (int): batch size
        wandb_wrapper (wandb_wrapper.WandbWrapper): initialized WandbWrapper
        dataset_conf (ml_collections.ConfigDict): dataset configuration
        optimizer_config (OptimizerConfig): optimizer configuration
        model_config (ml_collections.ConfigDict): model configuration
        total_kimg (int): total number of images to see (in thousands)
        report_loss_every_kimg (int): report loss every kimg
        save_ckpt_every_kimg (int): save checkpoint every kimg
        save_state_every_min (int): save last state every min (each snapshot overwrites the previous one)
        resume_training (bool): whether to resume training
        states_to_resume (str): path of the file where the saved states to resume are
        configs_to_resume (str): path of the file where the saved configurations to resume are
        seed (int): seed to use for reproducibility
        device (str): device to use
        loss_function (str): "edm" to use loss function from the paper EDM (Karras2022)
        use_ema (bool): whether to use the EMA of the model's parameters for sampling and saving NOT IMPLEMENTED
        ema_halflife_kimg (int): half-life of the EMA of the models weights in kimg
        ema_rampup_ratio (float): EMA rampup coeff (None->no rampup)
        check_ema_loss_every_kimg (int): check EMA loss every kimg
        use_pretrained_weights (bool): whether to use pretrained weights (eg from previous runs)
        pretrained_weights_path (str): path to the pretrained weights
        use_gradient_clipping (bool): whether to use gradient clipping
    """
    # if resume_training then the save file to resume must be provided
    assert (not resume_training) or (not use_pretrained_weights)
    if resume_training:
        assert states_to_resume is not None and configs_to_resume is not None
        assert os.path.exists(states_to_resume) and os.path.exists(configs_to_resume)
    if use_pretrained_weights:
        assert pretrained_weights_path is not None
        assert os.path.exists(pretrained_weights_path)
    # check inputs and set defaults
    assert optimizer_config.lr_schedule == "cosine"
    assert loss_function in ["edm"]

    # setup run directory and init variables
    os.makedirs(run_dir, exist_ok=True)
    samples_dir = os.path.join(run_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)
    checkpoints_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    total_nimg = total_kimg * 1000
    num_imgs_per_class = 10  # number of images to sample per class
    num_sampling_steps = 80

    # if resume_training, recover the configurations
    old_cur_nimg = None
    if resume_training:
        print("Resuming training from configs: %s", configs_to_resume)
        dataset_conf, model_config, optimizer_config, old_cur_nimg = (
            save_load.load_training_configs(configs_to_resume)
        )

    # set seed
    train_misc_utils.set_seeds(seed)

    # create model
    internal_model = ddpmpp.DDPMPP(**model_config)
    model = preconditioning.EDMPrecond(
        model=internal_model,
        sigma_data=0.5,  # value from the paper Karras2022
    )
    model.train().requires_grad_(True).to(device)

    # print model summary
    dummy_bs = batch_size
    torchinfo.summary(
        model,
        [
            (dummy_bs, 3, *dataset_conf.resolution),
            (dummy_bs,),
            (dummy_bs, model_config.label_dim),
        ],
        verbose=1,
    )

    # create dataloader
    dataloader = get_dataloaders.get_dataloader(batch_size, dataset_conf)

    # create optimizer
    optimizer = optim_utils.get_optimizer(optimizer_config, model)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=total_nimg // batch_size,
        eta_min=optimizer_config.lr_min,
    )

    # create loss function
    loss_fn = loss_utils.EDMLoss()

    # if resume training then load the model and optimizer and scheduler states
    if resume_training:
        print("Resuming training states from %s", states_to_resume)
        save_load.load_training_state(states_to_resume, model, optimizer, scheduler)
    else:
        if use_pretrained_weights:
            print(f"Loading pretrained weights from {pretrained_weights_path}")
            save_load.load_training_state(pretrained_weights_path, model)
            print("Pretrained weights loaded")
        else:
            print(
                "Not resuming training nor loading pretrained weights, starting from scratch."
            )

    if use_ema:
        ema_model = copy.deepcopy(model).eval().requires_grad_(False)

    # launch wandb run
    wandb_wrapper.init_run(
        config=dict(
            model_config=model_config.to_dict(),
            dataset_conf=dataset_conf.to_dict(),
            dataset_size=len(dataloader.dataset),
            optimizer_conf=optimizer_config.to_dict(),
            batch_size=batch_size,
            total_kimg=total_kimg,
            report_loss_every_kimg=report_loss_every_kimg,
            save_ckpt_every_kimg=save_ckpt_every_kimg,
            save_state_every_min=save_state_every_min,
            num_params=train_misc_utils.get_num_params(model),
            use_ema=use_ema,
            ema_halflife_kimg=ema_halflife_kimg,
            ema_rampup_ratio=ema_rampup_ratio,
            resume_training=resume_training,
            states_to_resume=states_to_resume,
            configs_to_resume=configs_to_resume,
            use_pretrained_weights=use_pretrained_weights,
            pretrained_weights_path=pretrained_weights_path,
            use_gradient_clipping=use_gradient_clipping,
        )
    )

    # training loop
    start_nimg = 0  # number of images seen at the start
    start_time = time.time()

    cur_nimg = old_cur_nimg if old_cur_nimg is not None else 0
    time_last_reported = time.time()
    last_reported_nimg = cur_nimg
    need_report_ema_loss = False
    last_checked_ema_loss_nimg = cur_nimg

    last_checkpoint_nimg = 0
    ckpt_idx = 1
    last_snapshot_time = time.time()

    loss_meter = train_misc_utils.AverageMeter("Loss")
    while cur_nimg < total_nimg:

        for batch in dataloader:
            # necessary because it might have been set to eval mode at the end of the loop
            model.requires_grad_(True).train()
            images, class_labels = train_misc_utils.prepare_batch(
                batch, device, dataset_conf.name
            )

            # compute loss and backpropagate
            optimizer.zero_grad()
            loss = loss_fn(model, images, class_labels)
            loss_meter.update(loss.item(), images.shape[0])
            loss.backward()

            if use_gradient_clipping:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0, norm_type=2.0
                )

            optimizer.step()
            scheduler.step()

            # update counters
            cur_nimg += images.shape[0]

            # update ema
            if use_ema:
                ema_halflife_nimg = ema_halflife_kimg * 1000
                if ema_rampup_ratio is not None:
                    ema_halflife_nimg = min(
                        ema_halflife_nimg, cur_nimg * ema_rampup_ratio
                    )
                ema_beta = 0.5 ** (batch_size / max(ema_halflife_nimg, 1e-8))
                for p_ema, p_mod in zip(ema_model.parameters(), model.parameters()):
                    # explanation : p_ema = p_ema * (1 - ema_beta) + p_mod * ema_beta
                    p_ema.copy_(p_mod.detach().lerp(p_ema, ema_beta))

                # check ema loss
                if (
                    cur_nimg - last_checked_ema_loss_nimg
                    >= check_ema_loss_every_kimg * 1000
                ):
                    with torch.inference_mode():
                        ema_model.eval()
                        ema_loss = loss_fn(ema_model, images, class_labels)
                    print(f"EMA loss: {ema_loss.item()}")
                    need_report_ema_loss = True
                    last_checked_ema_loss_nimg = cur_nimg

            # report loss
            if cur_nimg - last_reported_nimg >= report_loss_every_kimg * 1000:
                time_now = time.time()
                time_since_last_reported = time_now - time_last_reported
                nimg_since_last_reported = cur_nimg - last_reported_nimg
                est_time_remaining = (total_kimg * 1000 - cur_nimg) * (
                    (time_now - start_time) / (cur_nimg - start_nimg)
                )  # estimated time remaining
                print(
                    f"Cur_img {cur_nimg*1e-3:.2f}k: loss={loss_meter.avg},",
                    f"lr={optimizer.param_groups[0]['lr']}",
                )
                print(f"\tTime to process {nimg_since_last_reported} images:", end="")
                print(str(datetime.timedelta(seconds=time_since_last_reported)))
                print(
                    f"\testimated time remaining: {misc_utils.sec_to_dhms(est_time_remaining)}"
                )
                # compute and report grad norm
                if not use_gradient_clipping:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), float("inf"), norm_type=2.0
                    )
                print(
                    f"\tGrad norm: {grad_norm.item()}"  # pylint: disable=possibly-used-before-assignment
                )
                data_to_report = {
                    "train/loss": loss_meter.avg,
                    "nimg": cur_nimg,
                    "lr": optimizer.param_groups[0]["lr"],
                    "train/grad_norm": grad_norm.item(),
                }
                if need_report_ema_loss:
                    data_to_report["train/ema_loss"] = ema_loss.item()
                    need_report_ema_loss = False
                wandb_wrapper.log(data_to_report)
                loss_meter.reset()
                last_reported_nimg = cur_nimg
                time_last_reported = time_now
            else:  # we always report gradient norm
                if not use_gradient_clipping:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), float("inf"), norm_type=2.0
                    )
                data_to_report = {
                    "train/grad_norm": grad_norm.item(),
                    "nimg": cur_nimg,
                }
                wandb_wrapper.log(data_to_report)

            # save checkpoint
            if cur_nimg - last_checkpoint_nimg >= save_ckpt_every_kimg * 1000:
                print(f"Saving checkpoint {ckpt_idx}")
                ckpt_name = f"ckpt_{ckpt_idx}_{cur_nimg}img"
                ckpt_states_name = f"{ckpt_name}_states.pth"
                ckpt_config_name = f"{ckpt_name}_configs.pth"
                save_load.save_training_state(
                    path_to_save=os.path.join(checkpoints_dir, ckpt_states_name),
                    model=ema_model if use_ema else model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                )
                save_load.save_training_configs(
                    path_to_save=os.path.join(checkpoints_dir, ckpt_config_name),
                    dataset_config=dataset_conf,
                    model_config=model_config,
                    optim_config=optimizer_config,
                    cur_nimg=cur_nimg,
                )
                generate.sample_and_save(
                    path_to_save=os.path.join(samples_dir, f"{ckpt_name}.png"),
                    num_imgs_per_class=num_imgs_per_class,
                    model=ema_model if use_ema else model,
                    num_steps=num_sampling_steps,
                    device=device,
                )
                ckpt_idx += 1
                last_checkpoint_nimg = cur_nimg

            # save state (as last state)
            if time.time() - last_snapshot_time >= save_state_every_min * 60:
                print("Saving last snapshot")
                ckpt_states_name = "last_snapshot_states.pth"
                ckpt_config_name = "last_snapshot_configs.pth"
                save_load.save_training_state(
                    path_to_save=os.path.join(run_dir, ckpt_states_name),
                    model=ema_model if use_ema else model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                )
                save_load.save_training_configs(
                    path_to_save=os.path.join(run_dir, ckpt_config_name),
                    dataset_config=dataset_conf,
                    model_config=model_config,
                    optim_config=optimizer_config,
                    cur_nimg=cur_nimg,
                )
                generate.sample_and_save(
                    path_to_save=os.path.join(
                        samples_dir, f"snapshot_{cur_nimg}nimg.png"
                    ),
                    num_imgs_per_class=num_imgs_per_class,
                    model=ema_model if use_ema else model,
                    num_steps=num_sampling_steps,
                    device=device,
                )
                last_snapshot_time = time.time()
                print("Last snapshot saved")

    # save last state
    print("Saving last ema state")
    ckpt_name = f"end_training_{cur_nimg}img"
    ckpt_states_name = f"{ckpt_name}_states.pth"
    ckpt_config_name = f"{ckpt_name}_configs.pth"
    save_load.save_training_state(
        path_to_save=os.path.join(checkpoints_dir, ckpt_states_name),
        model=ema_model if use_ema else model,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    save_load.save_training_configs(
        path_to_save=os.path.join(checkpoints_dir, ckpt_config_name),
        dataset_config=dataset_conf,
        model_config=model_config,
        optim_config=optimizer_config,
        cur_nimg=cur_nimg,
    )
    generate.sample_and_save(
        path_to_save=os.path.join(samples_dir, f"{ckpt_name}.png"),
        num_imgs_per_class=num_imgs_per_class,
        model=ema_model if use_ema else model,
        num_steps=num_sampling_steps,
        device=device,
    )
    wandb_wrapper.stop_run()
