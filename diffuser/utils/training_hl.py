import os
import copy
import numpy as np
import torch
import einops
import pdb

from .arrays import batch_to_device, to_np, to_device, apply_dict
from .timer import Timer, Stat
from .cloud import sync_logs
from collections import defaultdict

def cycle(dl):
    while True:
        for data in dl:
            yield data

class EMA():
    '''
        empirical moving average
    '''
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class HierTrainer(object):
    def __init__(
        self,
        diffusion_model,
        dataset,
        val_dataset,
        renderer,
        ema_decay=0.995,
        train_batch_size=32,
        train_lr=2e-5,
        gradient_accumulate_every=2,
        step_start_ema=2000,
        update_ema_every=10,
        log_freq=1000,
        sample_freq=1000,
        save_freq=1000,
        label_freq=100000,
        save_parallel=False,
        results_folder='./results',
        n_reference=8,
        n_samples=2,
        bucket=None,
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.log_freq = log_freq
        self.sample_freq = sample_freq
        self.save_freq = save_freq
        self.label_freq = label_freq
        self.save_parallel = save_parallel

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.dataset = dataset
        self.val_dataset = val_dataset

        self.val_dataloader = torch.utils.data.DataLoader(
            self.val_dataset, batch_size=train_batch_size*4, num_workers=1, shuffle=True, pin_memory=True
        )
        self.dataloader = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=train_batch_size, num_workers=4, shuffle=True, pin_memory=True
        ))
        self.dataloader_vis = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=1, num_workers=0, shuffle=True, pin_memory=True
        ))
        self.renderer = renderer
        self.optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=train_lr)

        self.logdir = results_folder
        self.bucket = bucket
        self.n_reference = n_reference
        self.n_samples = n_samples

        self.reset_parameters()
        self.step = 0
        self.prev_hl_best = 10000000.
        self.prev_ll_best = 10000000.

        self.fourier_feature = dataset.fourier_feature

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    #-----------------------------------------------------------------------------#
    #------------------------------------ api ------------------------------------#
    #-----------------------------------------------------------------------------#
    def evaluate_nll(self, writer=None):

        timer = Timer()
        total_loss = 0.
        total_metrics = defaultdict(list)

        print('evaluating nll...')
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_dataloader):
                batch = next(self.dataloader)
                batch = batch_to_device(batch)

                metrics = self.model.calc_bpd_evaluation(*batch)

                for k, v in metrics.items():
                    total_metrics[k].append(v.detach().mean().item())


            batch_time = timer() / (batch_idx + 1)
            for k, v in total_metrics.items():
                total_metrics[k] = torch.stack(v, dim=0).mean()
            infos_str = ' | '.join([f'{key}: {val:8.4f}' for key, val in total_metrics.items()])
            print(f'{self.step}: {infos_str} | t: {batch_time:8.4f}', flush=True)
            writer.add_scalar('batch_time', batch_time, global_step=self.step )
            for k, v in total_metrics.items():
                writer.add_scalar('val_'+k, v, global_step=self.step )
            writer.flush()

    def comp_generalization_gap_obj(self, writer):

        train_losses = []
        val_losses = []
        for i in range(3):
            train_loss = self.evaluate(None, None, save_best=False, eval_n=None, split='train')
            train_losses.append(train_loss)
            val_loss = self.evaluate(None, None, save_best=False, eval_n=None, split='testing')
            val_losses.append(val_loss)
            print(f'train_loss: {train_loss}\t val_loss: {val_loss}')

        gaps = np.array(val_losses) - np.array(train_losses)
        print(f'train_losses: {train_losses}\t train_loss: {np.mean(train_losses)}\n')
        print(f'val_losses: {val_losses}\t val_loss: {np.mean(val_losses)}\n')
        print(f'gaps: {gaps}\t gap: {gaps.mean()}\n')

        writer.add_scalar('generalization_gap_per_dim', gaps.mean(), global_step=self.step )
        writer.add_scalar('epoch_train_loss/avg', np.mean(train_losses), global_step=self.step )
        writer.add_scalar('epoch_train_loss/std', np.std(train_losses), global_step=self.step )
        writer.add_scalar('epoch_val_loss/avg',np.mean( val_losses), global_step=self.step )
        writer.add_scalar('epoch_val_loss/std', np.std(val_losses), global_step=self.step )
        writer.flush()

    def evaluate(self, n_train_steps, writer=None, save_best=True, eval_n=None, split='testing'):

        timer = Timer()
        total_ll_loss = 0.
        total_hl_loss = 0.
        total_info = defaultdict(list)
        if split == 'train':
            dataloader = torch.utils.data.DataLoader(
                self.dataset, batch_size=self.batch_size*4, num_workers=1, shuffle=True, pin_memory=True
            )

        if split == 'testing':
            dataloader = torch.utils.data.DataLoader(
                self.val_dataset, batch_size=self.batch_size*4, num_workers=1, shuffle=True, pin_memory=True
            )

        print('evaluating...')
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                for i in range(self.gradient_accumulate_every):
                    batch = batch_to_device(batch)

                    loss, infos = self.model.loss(*batch, eval_n=None)
                    loss = loss / self.gradient_accumulate_every
                    ll_loss = infos['ll_s_loss'] + infos['ll_a_loss']
                    hl_loss = infos['hl_s_loss'] + infos['hl_a_loss']
                    ll_loss = ll_loss / self.gradient_accumulate_every
                    hl_loss = hl_loss / self.gradient_accumulate_every

                total_hl_loss += hl_loss
                total_ll_loss += ll_loss
                for k, v in infos.items():
                    if isinstance(v, torch.Tensor):
                        total_info[k].append(v.detach().item())
                    else:
                        total_info[k].append(v)

            batch_time = timer() / (batch_idx + 1)
            val_hl_loss = total_hl_loss / (batch_idx + 1)
            val_ll_loss = total_ll_loss / (batch_idx + 1)
            for k, v in total_info.items():
                total_info[k] = np.stack(v, axis=0).mean()
            infos_str = ' | '.join([f'{key}: {val:8.4f}' for key, val in total_info.items()])
            print(f'{self.step}: {val_hl_loss:8.4f} | {val_ll_loss:8.4f} | {infos_str} | t: {batch_time:8.4f}', flush=True)
            if writer:
                writer.add_scalar('val_ll_loss', val_ll_loss, global_step=self.step )
                writer.add_scalar('val_hl_loss', val_hl_loss, global_step=self.step )
                writer.add_scalar('batch_time', batch_time, global_step=self.step )
                for k, v in total_info.items():
                    writer.add_scalar('val_'+k, v, global_step=self.step )
                writer.flush()

            if val_hl_loss < self.prev_hl_best and save_best:
                self.prev_hl_best = val_hl_loss
                self.save(self.step, prefix='hl_best')
            if val_ll_loss < self.prev_ll_best and save_best:
                self.prev_ll_best = val_ll_loss
                self.save(self.step, prefix='ll_best')
        return val_hl_loss + val_ll_loss

    def train(self, n_train_steps, writer=None):

        timer = Timer()
        # forward_timer = Timer()
        # data_timer = Timer()
        # data_stat = Stat()
        # forward_stat = Stat()
        # backward_stat = Stat()
        # step_stat = Stat()
        for step in range(n_train_steps):
            for i in range(self.gradient_accumulate_every):

                # data_timer()
                batch = next(self.dataloader)
                batch = batch_to_device(batch)
                # data_end = data_timer()
                # data_mean = data_stat(data_end)


                # forward_timer()
                loss, infos = self.model.loss(*batch)
                # forward_time = forward_timer()
                # forward_mean = forward_stat(forward_time)

                # forward_timer()
                loss = loss / self.gradient_accumulate_every
                loss.backward()
                # backward_time = forward_timer()
                # backward_mean = backward_stat(backward_time)

            # forward_timer()
            self.optimizer.step()
            self.optimizer.zero_grad()
            # step_time = forward_timer()
            # step_mean = step_stat(step_time)

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step % self.save_freq == 0:
                label = self.step // self.label_freq * self.label_freq
                self.save(label)

            if self.step % self.log_freq == 0:
                infos_str = ' | '.join([f'{key}: {val:8.4f}' for key, val in infos.items()])
                batch_time = timer()
                print(f'{self.step}: {loss:8.4f} | {infos_str} | t: {batch_time:8.4f}')
                # print(f'runtime: data: {data_mean+forward_mean+backward_mean+step_mean: 8.4f}')
                # print(f'data: {data_mean:8.4f} | forward: {forward_mean:8.4f}  | backward: {backward_mean:8.4f} | step_time: {step_mean: 8.4f}')
                writer.add_scalar('total_loss', loss.detach().item(), global_step=self.step )
                writer.add_scalar('batch_time', batch_time, global_step=self.step )
                writer.add_scalars('infos', infos, global_step=self.step )
                writer.flush()

            if self.step == 0 and self.sample_freq:
                self.render_reference(self.n_reference)

            if self.sample_freq and self.step % self.sample_freq == 0:
                self.render_samples(n_samples=self.n_samples)

            self.step += 1

    def save(self, epoch, prefix=None):
        '''
            saves model and ema to disk;
            syncs to storage bucket if a bucket is specified
        '''
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict()
        }
        if prefix is not None:
            savepath = os.path.join(self.logdir, f'{prefix}_state.pt')
        else:
            savepath = os.path.join(self.logdir, f'state_{epoch}.pt')
        torch.save(data, savepath)
        print(f'[ utils/training ] Saved model to {savepath}')
        if self.bucket is not None:
            sync_logs(self.logdir, bucket=self.bucket, background=self.save_parallel)

    def load(self, epoch):
        '''
            loads model and ema from disk
        '''
        if isinstance(epoch, str) and 'best' in epoch:
            loadpath = os.path.join(self.logdir, f'{epoch}_state.pt')
        else:
            loadpath = os.path.join(self.logdir, f'state_{epoch}.pt')
        print(f'trainner load from {loadpath}')
        data = torch.load(loadpath)

        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])
    
    def load_ll_diffusion(self, epoch):

        if isinstance(epoch, str) and 'best' in epoch:
            loadpath = os.path.join(self.logdir, f'{epoch}_state.pt')
        else:
            loadpath = os.path.join(self.logdir, f'state_{epoch}.pt')
        print(f'trainner load from {loadpath}')
        data = torch.load(loadpath)

        self.step = data['step']

        named_paramd = dict()
        for n in data['model']:
            if n.startswith('ll_diffuser'):
                named_paramd[n[12:]] = data['model'][n]

        ema_named_paramd = dict()
        for n in data['ema']:
            if n.startswith('ll_diffuser'):
                ema_named_paramd[n[12:]] = data['ema'][n]

        self.model.ll_diffuser.load_state_dict(named_paramd)
        self.ema_model.ll_diffuser.load_state_dict(ema_named_paramd)

    def load_hl_diffusion(self, epoch):

        if isinstance(epoch, str) and 'best' in epoch:
            loadpath = os.path.join(self.logdir, f'{epoch}_state.pt')
        else:
            loadpath = os.path.join(self.logdir, f'state_{epoch}.pt')
        print(f'trainner load from {loadpath}')
        data = torch.load(loadpath)

        self.step = data['step']

        named_paramd = dict()
        for n in data['model']:
            if n.startswith('hl_diffuser'):
                named_paramd[n[12:]] = data['model'][n]

        ema_named_paramd = dict()
        for n in data['ema']:
            if n.startswith('hl_diffuser'):
                ema_named_paramd[n[12:]] = data['ema'][n]

        self.model.hl_diffuser.load_state_dict(named_paramd)
        self.ema_model.hl_diffuser.load_state_dict(ema_named_paramd)

    #-----------------------------------------------------------------------------#
    #--------------------------------- rendering ---------------------------------#
    #-----------------------------------------------------------------------------#

    def render_reference(self, batch_size=10):
        '''
            renders training points
        '''

        ## get a temporary dataloader to load a single batch
        dataloader_tmp = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=batch_size, num_workers=0, shuffle=True, pin_memory=True
        ))
        batch = dataloader_tmp.__next__()
        dataloader_tmp.close()

        ## get trajectories and condition at t=0 from batch
        trajectories = to_np(batch.trajectories)
        conditions = to_np(batch.conditions[0])[:,None]

        ## [ batch_size x horizon x observation_dim ]
        obs_dim = self.dataset.observation_dim
        act_dim = self.dataset.action_dim
        if self.fourier_feature:
            obs_dim = obs_dim//3
        normed_observations = trajectories[:, :, act_dim:act_dim+obs_dim]
        observations = self.dataset.normalizer.unnormalize(normed_observations, 'observations')

        # from diffusion.datasets.preprocessing import blocks_cumsum_quat
        # # observations = conditions + blocks_cumsum_quat(deltas)
        # observations = conditions + deltas.cumsum(axis=1)

        #### @TODO: remove block-stacking specific stuff
        # from diffusion.datasets.preprocessing import blocks_euler_to_quat, blocks_add_kuka
        # observations = blocks_add_kuka(observations)
        ####

        savepath = os.path.join(self.logdir, f'_sample-reference.png')
        self.renderer.composite(savepath, observations)

    def render_samples(self, batch_size=2, n_samples=2):
        '''
            renders samples from (ema) diffusion model
        '''
        for i in range(batch_size):

            ## get a single datapoint
            batch = self.dataloader_vis.__next__()

            conditions = to_device(batch.conditions, 'cuda:0')
            # hl diffussion
            x = to_device(batch.trajectories, 'cuda:0')
            B, H = x.shape[:2]
            hl_s = x[:, ::self.ema_model.jump, self.ema_model.action_dim:]
            hl_a = x[:, :, :self.ema_model.action_dim].reshape(B, H//self.ema_model.jump, self.ema_model.jump*self.ema_model.action_dim)
            hl_cond = {
              0: hl_s[:, 0],
              H // self.ema_model.jump - 1: hl_s[:, -1],
            }

            ## repeat each item in conditions `n_samples` times
            hl_cond = apply_dict(
                einops.repeat,
                hl_cond,
                'b d -> (repeat b) d', repeat=n_samples,
            )

            ## [ n_samples x horizon x (action_dim + observation_dim) ]
            hl_samples = self.ema_model.hl_diffuser.conditional_sample(hl_cond)
            # hl_samples = to_np(hl_samples)

            ## [ n_samples x horizon x observation_dim ]
            hl_state = hl_samples[:, :, self.model.hl_diffuser.action_dim:]

            # [ 1 x 1 x observation_dim ]
            normed_conditions = to_np(hl_cond[0])[:,None]

            ## [ n_samples x (horizon + 1) x observation_dim ]
            # if self.ema_model.hl_diffuser.condition:
            #     normed_observations = np.concatenate([
            #         np.repeat(normed_conditions, n_samples, axis=0),
            #         hl_state
            #     ], axis=1)

            ## [ n_samples x (horizon + 1) x observation_dim ]

            obs_dim = self.model.hl_diffuser.observation_dim
            act_dim = self.model.hl_diffuser.action_dim
            if self.fourier_feature:
                obs_dim = obs_dim//3
            observations = self.dataset.normalizer.unnormalize(to_np(hl_state[:,:,:obs_dim]), 'observations')

            savepath = os.path.join(self.logdir, f'hl_sample-{self.step}-{i}.png')
            self.renderer.composite(savepath, observations)

            hl_batch = torch.cat([hl_a, hl_s], dim=-1)
            hl_batch = einops.repeat(hl_batch, 'b h d -> (repeat b) h d', repeat=n_samples)
            _, _, hl_rec = self.ema_model.hl_diffuser.loss(hl_batch, hl_cond, return_rec=True)

            hl_rec_state = hl_rec[:, :, self.model.hl_diffuser.action_dim:]
            B, M = hl_rec_state.shape[:2]
            ll_cond_ = torch.stack([hl_rec_state[:, :-1], hl_rec_state[:, 1:]], dim=2)
            ll_cond_ = ll_cond_.reshape(B*(M-1), 2, -1)

            ll_cond = {
              0: ll_cond_[:, 0].detach(),
              self.ema_model.jump: ll_cond_[:, -1].detach(),
            }
            
            ll_samples = self.ema_model.ll_diffuser.conditional_sample(cond=ll_cond)
            ll_samples = ll_samples.reshape(B, (M-1), self.ema_model.jump+1, -1)
            ll_samples = torch.cat([ll_samples[:, 0, :1],
                          ll_samples[:, :, 1:].reshape(B, (M-1)*(self.ema_model.jump), -1)], dim=1)

            obs_dim = self.model.ll_diffuser.observation_dim
            act_dim = self.model.ll_diffuser.action_dim
            if self.fourier_feature:
                obs_dim = obs_dim//3
            normed_observations = ll_samples[:, :, act_dim:act_dim+obs_dim]

            ## [ n_samples x (horizon + 1) x observation_dim ]
            observations = self.dataset.normalizer.unnormalize(to_np(normed_observations), 'observations')

            savepath = os.path.join(self.logdir, f'll_hlrec_sample-{self.step}-{i}.png')
            self.renderer.composite(savepath, observations)

            M = hl_state.shape[1]
            ll_cond_ = torch.stack([hl_state[:, :-1], hl_state[:, 1:]], dim=2)
            ll_cond_ = ll_cond_.reshape(B*(M-1), 2, -1)

            ll_cond = {
              0: ll_cond_[:, 0].detach(),
              self.ema_model.jump: ll_cond_[:, -1].detach(),
            }
            
            ll_samples = self.ema_model.ll_diffuser.conditional_sample(cond=ll_cond)
            ll_samples = ll_samples.reshape(B, (M-1), self.ema_model.jump+1, -1)
            ll_samples = torch.cat([ll_samples[:, 0, :1],
                          ll_samples[:, :, 1:].reshape(B, (M-1)*(self.ema_model.jump), -1)], dim=1)
            ll_samples = to_np(ll_samples)
            normed_observations = ll_samples[:, :, act_dim:act_dim+obs_dim]

            ## [ n_samples x (horizon + 1) x observation_dim ]
            observations = self.dataset.normalizer.unnormalize(to_np(normed_observations), 'observations')

            savepath = os.path.join(self.logdir, f'll_hlgen_sample-{self.step}-{i}.png')
            self.renderer.composite(savepath, observations)

            # ll diffuser
            reshape = lambda x: x.reshape((1, ) + tuple([x.shape[1] // self.ema_model.jump, self.ema_model.jump]) + tuple(x.shape[2:]))
            ll_x = x[:, :-(self.ema_model.jump - 1)]
            ll_batch = torch.cat([reshape(ll_x[:, :-1]), ll_x[:, self.ema_model.jump::self.ema_model.jump][:, :, None]], axis=2)
            ll_batch = ll_batch.reshape(1 * (H // self.ema_model.jump-1), self.ema_model.jump+1, -1)
            ll_cond = {
              0: ll_batch[:, 0, self.ema_model.action_dim:],
              self.ema_model.jump: ll_batch[:, -1, self.ema_model.action_dim:]
            }
            ll_cond = apply_dict(
                einops.repeat,
                ll_cond,
                'b d -> (repeat b) d', repeat=n_samples,
            )
            ## [ n_samples x horizon x (action_dim + observation_dim) ]
            ll_samples = self.ema_model.ll_diffuser.conditional_sample(cond=ll_cond)
            ll_samples = ll_samples.reshape(B, (M-1), self.ema_model.jump+1, -1)
            ll_samples = torch.cat([ll_samples[:, 0, :1],
                          ll_samples[:, :, 1:].reshape(B, (M-1)*(self.ema_model.jump), -1)], dim=1)
            ll_samples = to_np(ll_samples)

            ## [ n_samples x horizon x observation_dim ]
            normed_observations = ll_samples[:, :, act_dim:act_dim+obs_dim]

            # [ 1 x 1 x observation_dim ]
            normed_conditions = to_np(ll_cond[0])[:,None]

            ## [ n_samples x (horizon + 1) x observation_dim ]
            # if self.ema_model.condition:
            #     normed_observations = np.concatenate([
            #         np.repeat(normed_conditions, n_samples, axis=0),
            #         normed_observations
            #     ], axis=1)

            ## [ n_samples x (horizon + 1) x observation_dim ]
            observations = self.dataset.normalizer.unnormalize(normed_observations, 'observations')

            savepath = os.path.join(self.logdir, f'll_ds_sample-{self.step}-{i}.png')
            self.renderer.composite(savepath, observations)
