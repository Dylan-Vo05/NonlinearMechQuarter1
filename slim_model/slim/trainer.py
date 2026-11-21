"""SlimTrainer: the repo's Trainer with a fixed training loop, plus optional distillation.

Subclasses the Trainer in this repo's denoising_diffusion_pytorch/ (that file is not edited).
What changes versus that Trainer (everything else is inherited unchanged):
  * train(): clean loop. The original resumes at the wrong pace (step advances twice per batch after
    load_model_step) and only saves a checkpoint at the very end; this one saves at every
    save_and_sample_every, resumes exactly, and can stop cleanly before a SLURM time limit.
  * save()/load(): only the main process writes, and weights are stored unwrapped (no 'module.'),
    so a checkpoint from 4 GPUs loads on 1 GPU and vice versa. load() is strict and raises.
  * Training and validation losses go to <run>/metrics.csv and stdout, not only to wandb.
  * Validation loss runs under torch.no_grad() (less memory), never includes the distillation term.
  * DataLoaders get num_workers (the original decodes every gif on the training process).
  * eval_network() is called at every checkpoint again (it is commented out in the repo's train()), and the
    results folder is kept relative to the working directory, because the repo's eval_network/save_preds
    build paths as './' + results_folder (an absolute /cosmos/... path would land in ./cosmos/... instead).
"""
import csv
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils import data
from accelerate.utils import broadcast

from denoising_diffusion_pytorch.video_denoising_diffusion_pytorch import GaussianDiffusion, Trainer, cycle, default

STOP_FOR_TIME = 99  # exit code train_slim.py uses when it stopped early to be resubmitted

# Teachers live outside the nn.Module tree so they are not wrapped by DDP, copied into the EMA
# model, or written into the student's checkpoint.
_TEACHERS = {}


class DistillDiffusion(GaussianDiffusion):
    """GaussianDiffusion whose training loss adds ||eps_student - eps_teacher||^2 (output distillation).

    Teacher and student see the same noisy input, timestep, conditioning and the same
    classifier-free-guidance null mask (the RNG is forked so both draw the same mask).
    """

    def __init__(self, *args, distill_weight=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.distill_weight = float(distill_weight)
        self.distill_active = True
        self._teacher_key = None

    def set_teacher(self, teacher):
        teacher.eval().requires_grad_(False)
        assert teacher.num_timesteps == self.num_timesteps, 'teacher and student need the same noise schedule'
        assert teacher.channels == self.channels and teacher.num_frames == self.num_frames
        self._teacher_key = id(teacher)
        _TEACHERS[self._teacher_key] = teacher

    def p_losses(self, x_start, t, cond=None, noise=None, **kwargs):
        teacher = _TEACHERS.get(self._teacher_key)
        if teacher is None or not self.distill_active or not torch.is_grad_enabled():
            return super().p_losses(x_start, t, cond=cond, noise=noise, **kwargs)

        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        if next(teacher.parameters()).device != x_noisy.device:
            teacher.to(x_noisy.device)
        devices = [x_noisy.device.index if x_noisy.device.index is not None else torch.cuda.current_device()] \
            if x_noisy.is_cuda else []
        with torch.no_grad(), torch.random.fork_rng(devices=devices):
            eps_teacher = teacher.denoise_fn(x_noisy, t, cond=cond, **kwargs)
        eps_student = self.denoise_fn(x_noisy, t, cond=cond, **kwargs)

        loss = F.l1_loss(noise, eps_student)  # the authors' loss_type='l1'
        return loss + self.distill_weight * F.mse_loss(eps_student.float(), eps_teacher.float())


class SlimTrainer(Trainer):

    def __init__(self, *args, num_workers=4, log_every=100, time_budget_hours=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        # The parent's eval_network/save_preds write to './' + results_folder. With an absolute path that becomes
        # './/cosmos/...', i.e. a folder under the working directory. A relative path makes './' harmless.
        self.results_folder = Path(os.path.relpath(os.path.abspath(str(self.results_folder))))
        self.log_every = int(log_every)
        self.time_budget_s = float(time_budget_hours) * 3600.0
        self.metrics_path = os.path.join(str(self.results_folder), 'metrics.csv')

        if num_workers > 0:
            mk = dict(num_workers=num_workers, pin_memory=True, persistent_workers=True)
            self.dl = cycle(self.accelerator.prepare(
                data.DataLoader(self.ds, batch_size=self.batch_size, shuffle=True, drop_last=True, **mk)))
            self.dl_test = self.accelerator.prepare(
                data.DataLoader(self.ds_test, batch_size=self.test_batch_size, shuffle=False, **mk))

    # ---------- logging ----------
    def _write_metric(self, step, name, value):
        self.log_fn({name: value}, step=step)
        if not self.accelerator.is_main_process:
            return
        new = not os.path.exists(self.metrics_path)
        with open(self.metrics_path, 'a', newline='') as f:
            w = csv.writer(f)
            if new:
                w.writerow(['step', 'metric', 'value', 'unix_time'])
            w.writerow([step, name, f'{value:.6g}', f'{time.time():.0f}'])

    # ---------- checkpoints ----------
    def _unwrapped(self):
        return self.accelerator.unwrap_model(self.model), self.accelerator.unwrap_model(self.ema_model)

    def save(self, step=None):
        step = self.step if step is None else step
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            model, ema = self._unwrapped()
            save_dir = os.path.join(str(self.results_folder), 'model', f'step_{step}')
            os.makedirs(save_dir, exist_ok=True)
            tmp = os.path.join(save_dir, 'checkpoint.pt.tmp')
            torch.save(dict(model=model.state_dict(), ema=ema.state_dict(),
                            optimizer=self.opt.state_dict(), steps=step), tmp)
            os.replace(tmp, os.path.join(save_dir, 'checkpoint.pt'))  # atomic: never a half-written file
            print(f'checkpoint saved: {save_dir}/checkpoint.pt', flush=True)
        self.accelerator.wait_for_everyone()

    def load(self, strict=True):
        from slim.common import clean_state_dict, torch_load
        path = os.path.join(str(self.results_folder), 'model', f'step_{self.step}', 'checkpoint.pt')
        obj = torch_load(path)
        model, ema = self._unwrapped()
        model.load_state_dict(clean_state_dict(obj['model']), strict=strict)
        ema.load_state_dict(clean_state_dict(obj.get('ema', obj['model'])), strict=strict)
        try:
            self.opt.load_state_dict(obj['optimizer'])
        except Exception as e:  # e.g. resuming the authors' checkpoint with a new optimizer
            self.accelerator.print(f'optimizer state not restored ({e}); continuing with a fresh optimizer')
        self.accelerator.print(f'resumed from {path}')
        return obj

    # ---------- validation ----------
    def eval_network(self, *args, **kwargs):
        model, ema = self._unwrapped()
        flags = [(m, getattr(m, 'distill_active', None)) for m in (model, ema)]
        for m, _ in flags:
            if hasattr(m, 'distill_active'):
                m.distill_active = False
        # capture the validation loss the parent logs, so it also lands in metrics.csv
        captured = {}
        orig_log = self.log_fn

        def tap(values, step=None):
            captured.update(values)
            orig_log(values, step=step)

        self.log_fn = tap
        try:
            with torch.no_grad():
                super().eval_network(*args, **kwargs)
        finally:
            self.log_fn = orig_log
            for m, v in flags:
                if v is not None:
                    m.distill_active = v
        if 'validation loss' in captured:
            self._write_metric(self.step, 'val_loss', float(captured['validation loss']))
            self.accelerator.print(f'step {self.step}: validation loss {captured["validation loss"]:.5f}', flush=True)

    # ---------- training ----------
    def _should_stop(self, t0):
        flag = torch.zeros(1, device=self.device)
        if self.accelerator.is_main_process and self.time_budget_s > 0 and time.time() - t0 > self.time_budget_s:
            flag += 1
        if self.accelerator.num_processes > 1:
            flag = broadcast(flag, from_process=0)
        return bool(flag.item() > 0)

    def train(self, resume_step=None, num_samples=3, num_preds=1, prob_focus_present=0., focus_present_mask=None):
        """Returns 0 when train_num_steps is reached, STOP_FOR_TIME when it stopped for the time budget."""
        if resume_step is not None:
            self.step = int(resume_step)
            self.load()
            if self.step >= self.train_num_steps:
                self.accelerator.print(f'already trained to step {self.step}; nothing to do')
                return 0

        t0 = time.time()
        t_last, loss_sum, loss_n = time.time(), 0.0, 0
        self.model.train()
        while self.step < self.train_num_steps:
            x, cond = next(self.dl)
            self.opt.zero_grad(set_to_none=True)
            loss = self.model(x=x, cond=cond, null_cond_prob=self.null_cond_prob,
                              prob_focus_present=prob_focus_present, focus_present_mask=focus_present_mask)
            self.accelerator.backward(loss)
            if self.max_grad_norm is not None and self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.opt.step()
            self.step += 1

            loss_sum += loss.item()
            loss_n += 1
            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step % self.log_every == 0:
                avg = loss_sum / max(loss_n, 1)
                rate = loss_n / max(time.time() - t_last, 1e-9)
                self._write_metric(self.step, 'train_loss', avg)
                self.accelerator.print(f'step {self.step}/{self.train_num_steps}  train loss {avg:.5f}  '
                                       f'{rate:.2f} it/s  elapsed {(time.time() - t0) / 3600:.2f} h', flush=True)
                t_last, loss_sum, loss_n = time.time(), 0.0, 0
                if self._should_stop(t0):
                    self.save()
                    self.accelerator.print(f'time budget reached at step {self.step}; saved, exiting to be resumed')
                    self.accelerator.end_training()
                    return STOP_FOR_TIME

            if self.step % self.save_and_sample_every == 0 or self.step == self.train_num_steps:
                self.save()
                if self.step % self.save_and_sample_every == 0:
                    # the parent splits num_samples * num_preds videos across GPUs and fails if a GPU gets none
                    preds = max(num_preds, -(-self.accelerator.num_processes // num_samples))
                    self.eval_network(prob_focus_present, focus_present_mask, num_samples=num_samples, num_preds=preds)
                self.model.train()

        self.accelerator.print('training completed')
        self.accelerator.end_training()
        return 0
