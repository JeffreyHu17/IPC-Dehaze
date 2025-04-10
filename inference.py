import os
from tqdm import tqdm
from basicsr.utils import imwrite, tensor2img,img2tensor
import argparse
import cv2
import glob
import os
import torch
from basicsr.utils import parallel_decode_my
from basicsr.archs.dehazeToken_arch import DehazeTokenNet,Critic

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictor_path',type=str,default='pretrained_models/predictor.pth')
    parser.add_argument('--critic_path',type=str,default=  'pretrained_models/critic.pth')
    parser.add_argument('-i','--input', type=str, default='examples', help='input test image folder')
    parser.add_argument('-o','--output', type=str, default='results', help='output test image folder')
    parser.add_argument('-n', type=int, default=8, help='num_iterations')
    parser.add_argument('--max_size', type=int, default=1500, help='max_size')
    args = parser.parse_args()

    num_iterations = args.n
    os.makedirs(args.output, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net_g = DehazeTokenNet(codebook_params=[64, 1024, 256], blk_depth=16, LQ_stage=True, predictor_name='swinLayer').to(device)
    net_g.load_state_dict(torch.load(args.predictor_path)['params'], strict=True)
    net_g.eval()
    
    net_critic= Critic().to(device)
    net_critic.load_state_dict(torch.load(args.critic_path)['params'], strict=True)
    net_critic.eval()

    def tokens_to_logits(seq: torch.Tensor,h=0,w=0,critic=False) -> torch.Tensor:
        if critic:
            logits=net_critic(seq,h,w)
        else:
            logits = net_g.transformer(seq,critic)
        return logits 
    
    def tokens_to_feats(seq:torch.Tensor)->torch.Tensor:
        '''
        seq :b, 1, h, w
        '''
        feats = net_g.vqgan.quantize.get_codebook_entry(seq.long())
        return feats

    if os.path.isfile(args.input):
        paths = [args.input]
    else:
        paths = sorted(glob.glob(os.path.join(args.input, '*')))
   
    pbar = tqdm(total=len(paths), unit='image')
    for idx, path in enumerate(paths):
        imgname = os.path.splitext(os.path.basename(path))[0]
        pbar.set_description(f'Test {idx} {imgname}')
        img = cv2.imread(path, cv2.IMREAD_COLOR)/255.0
        img=img2tensor(img)
        img = img.unsqueeze(0).to(device)

        with torch.no_grad():
            _,_,origin_h,origin_w = img.shape

            if origin_h*origin_w >= args.max_size**2:
                scale = args.max_size/max(origin_h, origin_w)
                img = torch.nn.UpsamplingBilinear2d(scale_factor=scale)(img)
    
            # padding to multiple of window_size * 8
            wsz = 32
            _, _, h_old, w_old = img.shape
            h_pad = (h_old // wsz + 1) * wsz - h_old
            w_pad = (w_old // wsz + 1) * wsz - w_old
            img = torch.cat([img, torch.flip(img, [2])],
                            2)[:, :, :h_old + h_pad, :]
            img = torch.cat([img, torch.flip(img, [3])],
                            3)[:, :, :, :w_old + w_pad]
    
            ################### Encoder #####################
            enc_feats = net_g.vqgan.multiscale_encoder(img.detach())
            enc_feats = enc_feats[::-1]

            x = enc_feats[0]
            b, c, h, w = x.shape

            feat_to_quant = net_g.vqgan.before_quant(x)
            _,_,lq_tokens=net_g.vqgan.encode(img)
            
            mask_tokens=-1*torch.ones(b,h*w).to(feat_to_quant.device).long()

            output_tokens,mask_tokens = parallel_decode_my.decode_critic_only(
                mask_tokens,
                feat_to_quant,
                tokens_to_logits,
                tokens_to_feats,
                num_iter=num_iterations,
                )
            
            quant_feats=tokens_to_feats(output_tokens[:, -1,:].reshape(b,1,h,w))

                # ################## Generator ####################
            after_quant_feat = net_g.vqgan.after_quant(quant_feats)

            x=after_quant_feat
            for i in range(net_g.max_depth):
                cur_res = net_g.gt_res // 2**net_g.max_depth * 2**i
                x = net_g.fuse_convs_dict[str(cur_res)](enc_feats[i].detach(), x, 1)
                x = net_g.vqgan.decoder_group[i](x)
                
            output_img = net_g.vqgan.out_conv(x)
            output_img = output_img[..., :h_old , :w_old ]
            
            if origin_h*origin_w>=args.max_size**2:
                output_img = torch.nn.UpsamplingBilinear2d((origin_h, origin_w))(output_img)
                
            cv2.imwrite(os.path.join(args.output, f'{imgname}.png'),tensor2img(output_img,rgb2bgr=True))


        pbar.update(1)
           
    pbar.close()


if __name__ == '__main__':
    main()
