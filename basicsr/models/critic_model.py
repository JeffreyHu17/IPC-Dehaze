import torch
from collections import OrderedDict
from os import path as osp
import os
from basicsr.utils import img2tensor, tensor2img, imwrite
from tqdm import tqdm
from basicsr.data.transforms import paired_random_crop
from basicsr.archs import build_network
from basicsr.losses import build_loss
from .base_model import BaseModel
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
import torchvision.utils as tvu
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.checkpoint import checkpoint
import cv2
from basicsr.utils.mask_schedule import schedule
import math
from ..utils.parallel_decode import gumbel_sample
@MODEL_REGISTRY.register()
class CriticModel(BaseModel):
    """Base UnderwaterNet model for single image super-resolution."""
    def __init__(self, opt):
        super().__init__(opt)
        self.queue_size = opt.get('queue_size', 180)

        # define network g
        self.net_g = build_network(self.opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)
        
        # define network critic
        self.net_critic = build_network(self.opt['network_critic'])
        self.net_critic = self.model_to_device(self.net_critic)
        self.print_network(self.net_critic)
        
        # define network hq
        self.net_hq = build_network(self.opt['network_hq'])
        self.net_hq = self.model_to_device(self.net_hq)
        self.print_network(self.net_hq)

        load_path = self.opt['path'].get('pretrain_network_hq', None)
        if load_path is not None:
            self.load_network(self.net_hq, load_path,
                              self.opt['path'].get('strict_load', True))

        
        # load pre-trained HQ ckpt, frozen decoder and codebook
        self.LQ_stage = self.opt['network_g'].get('LQ_stage', False)
        if self.LQ_stage:
            load_path = self.opt['path'].get('pretrain_network_hq', None)
            self.load_network(self.net_g.module.vqgan, load_path, False)
            frozen_module_keywords = self.opt['network_g'].get(
                'frozen_module_keywords', None)
            if frozen_module_keywords is not None:
                for name, module in self.net_g.named_modules():
                    for fkw in frozen_module_keywords:
                        if fkw in name:
                            # print(fkw)
                            for p in module.parameters():
                                p.requires_grad = False
                            break

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        logger = get_root_logger()
        if load_path is not None:
            logger.info(f'Loading net_g from {load_path}')
            self.load_network(self.net_g, load_path,
                              self.opt['path']['strict_load'])

        if self.is_train:
            self.init_training_settings()

    def feed_data(self, data):
        if 'lq' in data:
            self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_critic.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_critic'].pop('type')
        self.optimizer_critic = self.get_optimizer(optim_type, optim_params,
                                              **train_opt['optim_critic'])
        self.optimizers.append(self.optimizer_critic)

    @torch.no_grad()
    def _dequeue_and_enqueue(self):
        """It is the training pair pool for increasing the diversity in a batch.

        Batch processing limits the diversity of synthetic degradations in a batch. For example, samples in a
        batch could not have different resize scaling factors. Therefore, we employ this training pair pool
        to increase the degradation diversity in a batch.
        """
        # initialize
        b, c, h, w = self.lq.size()
        if not hasattr(self, 'queue_lr'):
            assert self.queue_size % b == 0, f'queue size {self.queue_size} should be divisible by batch size {b}'
            self.queue_lr = torch.zeros(self.queue_size, c, h, w).cuda()
            _, c, h, w = self.gt.size()
            self.queue_gt = torch.zeros(self.queue_size, c, h, w).cuda()
            self.queue_ptr = 0
        if self.queue_ptr == self.queue_size:  # the pool is full
            # do dequeue and enqueue
            # shuffle
            idx = torch.randperm(self.queue_size)
            self.queue_lr = self.queue_lr[idx]
            self.queue_gt = self.queue_gt[idx]
            # get first b samples
            lq_dequeue = self.queue_lr[0:b, :, :, :].clone()
            gt_dequeue = self.queue_gt[0:b, :, :, :].clone()
            # update the queue
            self.queue_lr[0:b, :, :, :] = self.lq.clone()
            self.queue_gt[0:b, :, :, :] = self.gt.clone()

            self.lq = lq_dequeue
            self.gt = gt_dequeue
        else:
            # only do enqueue
            self.queue_lr[self.queue_ptr:self.queue_ptr +
                          b, :, :, :] = self.lq.clone()
            self.queue_gt[self.queue_ptr:self.queue_ptr +
                          b, :, :, :] = self.gt.clone()
            self.queue_ptr = self.queue_ptr + b

    def init_training_settings(self):
        self.net_hq.eval()
        self.net_g.eval()
        self.net_critic.train()

        train_opt = self.opt['train']

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(
                self.device)
        else:
            self.cri_perceptual = None

        if train_opt.get('feats_opt'):
            self.cri_feats = build_loss(train_opt['feats_opt']).to(self.device)
        else:
            self.cri_feats = None

        if self.cri_pix is None and self.cri_perceptual is None and self.cri_feats is None:
            print('Both pixel, perceptual, and feats losses are None.')
            pass
            # raise ValueError('Both pixel and perceptual losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def get_current_visuals(self):
        vis_samples = 4
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()[:vis_samples]
        if hasattr(self, 'output'):
            out_dict['output'] = self.output.detach().cpu()[:vis_samples]
            # cv2.imwrite("output.jpg", tensor2img(self.output[0]))
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()[:vis_samples]
        if hasattr(self, 'hq_rec'):
            out_dict['hq_rec'] = self.hq_rec.detach().cpu()[:vis_samples]

        return out_dict
    def tokens_to_feats(self,gt_indices):
        return self.net_hq.module.quantize.get_codebook_entry(
            gt_indices)
    def optimize_parameters(self, current_iter):

        train_opt = self.opt['train']
        code_only=self.opt['network_g']['code_only']
        detach_16=self.opt['network_g']['detach_16']

        self.optimizer_critic.zero_grad()
        # target_feats,result_dict=self.net_hq.encode(self.gt)
        if self.LQ_stage:
            with torch.no_grad():
                self.hq_rec, _, _, gt_indices = self.net_hq(self.gt)
                b, _, h, w = gt_indices.shape

                target_feats = self.net_hq.module.quantize.get_codebook_entry(
                    gt_indices)

                # mask
                r = np.random.uniform()
                r = math.floor(schedule(r, h * w) * h * w)
                
                samples = torch.rand(b,h * w,dtype=torch.float,device=self.device).topk(r,dim=1).indices
            
                token_mask = torch.zeros(b,h * w,device=self.device,dtype=torch.bool)
                token_mask.scatter_(dim=1, index=samples, value=True)
                token_mask = token_mask.reshape(b, 1, h, w)
                
                logits,lq_feats = self.net_g(
                    self.lq, hq_feats=target_feats,token_mask=token_mask,
                    code_only=code_only)  # logits (hw)bn -> b(hw)n

        l_g_total = 0
        loss_dict = OrderedDict()
        
    
        # todo temperature
        logits/=2
        probs = F.softmax(logits, -1)


        pred_ids = torch.multinomial(probs.view(b*h*w, -1), 1)
        pred_ids = pred_ids.view(b, h*w)
    
        critic_labels = (pred_ids!=gt_indices.flatten(1)).float()
        
        critic_logits = self.net_critic.forward(pred_ids,h,w)
        
        l_critic = F.binary_cross_entropy_with_logits(
            critic_logits,
            critic_labels) 
        
        loss_dict['l_critic'] = l_critic
        l_g_total += l_critic


        l_g_total.backward()
        self.optimizer_critic.step()


        self.log_dict = self.reduce_loss_dict(loss_dict)
        
    def test(self):
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(self.lq)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(self.lq)
            self.net_g.train()

    def test_selfensemble(self):
        # TODO: to be tested
        # 8 augmentations
        # modified from https://github.com/thstkdgus35/EDSR-PyTorch

        def _transform(v, op):
            # if self.precision != 'single': v = v.float()
            v2np = v.data.cpu().numpy()
            if op == 'v':
                tfnp = v2np[:, :, :, ::-1].copy()
            elif op == 'h':
                tfnp = v2np[:, :, ::-1, :].copy()
            elif op == 't':
                tfnp = v2np.transpose((0, 1, 3, 2)).copy()

            ret = torch.Tensor(tfnp).to(self.device)
            # if self.precision == 'half': ret = ret.half()

            return ret

        # prepare augmented data
        lq_list = [self.lq]
        for tf in 'v', 'h', 't':
            lq_list.extend([_transform(t, tf) for t in lq_list])

        # inference
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                out_list = [self.net_g_ema(aug) for aug in lq_list]
        else:
            self.net_g.eval()
            with torch.no_grad():
                out_list = [self.net_g_ema(aug) for aug in lq_list]
            self.net_g.train()

        # merge results
        for i in range(len(out_list)):
            if i > 3:
                out_list[i] = _transform(out_list[i], 't')
            if i % 4 > 1:
                out_list[i] = _transform(out_list[i], 'h')
            if (i % 4) % 2 == 1:
                out_list[i] = _transform(out_list[i], 'v')
        output = torch.cat(out_list, dim=0)

        self.output = output.mean(dim=0, keepdim=True)
    def nondist_validation(self, dataloader, current_iter, tb_logger,
                           save_img, save_as_dir):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }
            
        pbar = tqdm(total=len(dataloader), unit='image')

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)

            # zero self.metric_results
            self.metric_results = {metric: 0 for metric in self.metric_results}
            self.key_metric = self.opt['val'].get('key_metric') 

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test()
            
            sr_img = tensor2img(self.output)
            metric_data = [img2tensor(sr_img).unsqueeze(0) / 255, self.gt]

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], 'image_results',
                                             f'{current_iter}', 
                                             f'{img_name}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(
                            self.opt['path']['visualization'], dataset_name,
                            f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(
                            self.opt['path']['visualization'], dataset_name,
                            f'{img_name}_{self.opt["name"]}.png')
                if save_as_dir:
                    save_as_img_path = osp.join(save_as_dir, f'{img_name}.png')
                    imwrite(sr_img, save_as_img_path)
                imwrite(sr_img, save_img_path)

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    tmp_result = self.metric_funcs[name](*metric_data)
                    self.metric_results[name] += tmp_result.item() 

            pbar.update(1)
            pbar.set_description(f'Test {img_name}')

        pbar.close()
            
        if with_metrics:
            # calculate average metric
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
            
            if self.key_metric is not None:
                # If the best metric is updated, update and save best model
                to_update = self._update_best_metric_result(dataset_name, self.key_metric, self.metric_results[self.key_metric], current_iter)
            
                if to_update:
                    for name, opt_ in self.opt['val']['metrics'].items():
                        self._update_metric_result(dataset_name, name, self.metric_results[name], current_iter)
                    self.copy_model(self.net_g, self.net_g_best)
                    self.copy_model(self.net_d, self.net_d_best)
                    self.save_network(self.net_g, 'net_g_best', '')
                    self.save_network(self.net_d, 'net_d_best', '')
            else:
                # update each metric separately 
                updated = []
                for name, opt_ in self.opt['val']['metrics'].items():
                    tmp_updated = self._update_best_metric_result(dataset_name, name, self.metric_results[name], current_iter)
                    updated.append(tmp_updated)
                # save best model if any metric is updated 
                if sum(updated): 
                    self.copy_model(self.net_g, self.net_g_best)
                    self.copy_model(self.net_d, self.net_d_best)
                    self.save_network(self.net_g, 'net_g_best', '')
                    self.save_network(self.net_d, 'net_d_best', '')

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)
            
    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)
    
    def vis_single_code(self, up_factor=2):
        net_g = self.get_bare_model(self.net_g)
        codenum = self.opt['network_g']['codebook_params'][0][1]
        with torch.no_grad():
            code_idx = torch.arange(codenum).reshape(codenum, 1, 1, 1)
            code_idx = code_idx.repeat(1, 1, up_factor, up_factor)
            output_img = net_g.decode_indices(code_idx) 
            output_img = tvu.make_grid(output_img, nrow=32)

        return output_img.unsqueeze(0)
    def save(self, epoch, current_iter):
        self.save_network(self.net_g, 'net_g', current_iter)
        self.save_network(self.net_critic, 'net_critic', current_iter)
        # self.save_network(self.net_d, 'net_d', current_iter)
        self.save_training_state(epoch, current_iter)
