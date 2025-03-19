import os
from tqdm import tqdm
from basicsr.utils import imwrite, tensor2img,img2tensor
import argparse
import cv2
import glob
from PIL import Image
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from basicsr.utils import parallel_decode_my
from basicsr.archs.dehazeToken_arch import DehazeTokenNet,Critic

# # for plotting
plt.rcParams["figure.figsize"] = (12, 12)  # set default size of plots
def get_color(value):
    if value == 1:
        return (219, 219, 223)  # 颜色为（219，219，223）
    elif value == 0:
        return (175, 175, 188)  # 颜色为（175，175，188）
    else:
        return 'white'  # 其他情况为白色
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictor_path',type=str,default='pretrained_models/net_predictor.pth')
    parser.add_argument('--critic_path',type=str,default=  'pretrained_models/net_critic.pth')
    parser.add_argument('-i','--input', type=str, default='examples', help='input test image folder')
    parser.add_argument('-o','--output', type=str, default='results', help='input test image folder')
    parser.add_argument('--show_middle', action="store_true", help='show the middle result')
    parser.add_argument('--referrence', type=str, default='./datasets/outdoor/gt', help='input test image folder')
    parser.add_argument('-n', type=int, default=8, help='num_iterations')
    args = parser.parse_args()

    num_iterations = args.n

    # tem = "URHI"
    parallel_folder=f'{args.output}/parallel'
    mask_folder=f'{args.output}/mask'
    img_folder=f'{args.output}/img'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net_g = DehazeTokenNet(codebook_params=[64, 1024, 256], blk_depth=16,LQ_stage=True,predictor_name='swinLayer').to(device)

    net_g.load_state_dict(torch.load(args.predictor_path)['params'], strict=True)
    net_g.eval()
    
    net_critic= Critic().to(device)
    net_critic.load_state_dict(torch.load(args.critic_path)['params'], strict=True)
    net_critic.eval()
    
    os.makedirs(parallel_folder, exist_ok=True)
    os.makedirs(mask_folder, exist_ok=True)
    os.makedirs(img_folder, exist_ok=True)

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
         
            max_size = 800
            if origin_h*origin_w >= max_size**2:
                scale = max_size/max(origin_h, origin_w)
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
            # logits=net_g.transformer(torch.cat((lq_feats,lq_feats,t_map),dim=1))

            ################# Quantization ###################
            # out_tokens=logits.argmax(dim=2)
            # quant_feats=net_g.vqgan.quantizer.decode_ids(out_tokens.reshape(x.shape[0],h,w))
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
            
            if origin_h*origin_w>=max_size**2:
                output_img = torch.nn.UpsamplingBilinear2d((origin_h, origin_w))(output_img)
                
            # prev_mask =torch.ones_like(mask_tokens[:, 0, :])
            cv2.imwrite(os.path.join(img_folder, f'{imgname}.png'),tensor2img(output_img,rgb2bgr=True))
            '''
            plot mask together
                            mask_tokens=mask_tokens.squeeze(0).reshape(num_iterations,h,w).cpu()
            down=torch.nn.UpsamplingBilinear2d((h, w))(output_img)
            combined_image=visualize_masks_with_color(tensor2img(down,rgb2bgr=False),mask_tokens,2,4)
            
            # 合并处理后的掩码图
            
            # combined_mask = np.bitwise_or.reduce(combined_image, axis=0)
            plt.imshow(combined_image, cmap='gray')
            plt.axis('off')
            plt.savefig(os.path.join(mask_folder, f'{imgname}.png'),bbox_inches="tight")  
            
            '''
            
            # 设置图像的固定分辨率
        
            mask_tokens_cpu = mask_tokens.squeeze(0).reshape(num_iterations, h, w).cpu()

            # # 定义颜色映射函数


            # # 循环遍历每个iteration
            # for i in range(num_iterations):
            #     # 获取当前iteration的mask
            #     mask = mask_tokens_cpu[i]
                
            #     # 将mask tensor转换为NumPy数组并绘制图像
            #     mask_image = mask.numpy()
                
            #     # 创建图像数组并根据值设置颜色
            #     colored_mask_image = np.zeros((mask_image.shape[0]*4, mask_image.shape[1]*4, 3), dtype=np.uint8)
            #     for x in range(mask_image.shape[0]):
            #         for y in range(mask_image.shape[1]):
            #             color = get_color(mask_image[x, y])
            #             colored_mask_image[x*4:(x+1)*4, y*4:(y+1)*4] = color
                
            #     # 将 NumPy 数组转换为 PIL 图像对象
            #     image = Image.fromarray(colored_mask_image)
                
            #     # 设置图像的分辨率并保存图像
            #     # image = image.resize((W, H), resample=Image.LANCZOS)
            #     image.save(os.path.join(mask_folder, f'{imgname}_{i}.png'))
            if args.show_middle:
            
                for i in range(num_iterations-1):
                    mask = mask_tokens_cpu[i]
                
                    # 将mask tensor转换为NumPy数组并绘制图像
                    mask_image = mask.numpy()
                    
                    # 创建图像数组并根据值设置颜色
                    colored_mask_image = np.zeros((mask_image.shape[0]*4, mask_image.shape[1]*4, 3), dtype=np.uint8)
                    for x in range(mask_image.shape[0]):
                        for y in range(mask_image.shape[1]):
                            color = get_color(mask_image[x, y])
                            colored_mask_image[x*4:(x+1)*4, y*4:(y+1)*4] = color
                    
                    # 将 NumPy 数组转换为 PIL 图像对象
                    image = Image.fromarray(colored_mask_image)

                    image.save(os.path.join(mask_folder, f'{imgname}_{i}.png'))
            
                    # fig_para=plt.figure(figsize=(12, 8))
                    # ax_para=fig_para.add_subplot(2,4,i+1)
                    # unpredicted tokens will be replaced by low quality tokens
                    # 定义深灰色的颜色，带有透明度
            
                    
                    tokens = output_tokens[:, i,:].reshape(b,1,h,w)*(1-mask_tokens[:, i+1,:]).reshape(b,1,h,w)+lq_tokens.reshape(b,1,h,w)*(mask_tokens[:, i+1,:].reshape(b,1,h,w))
                    # quant_feats=tokens_to_feats(output_tokens[:, i,:].reshape(b,1,h,w))
                    quant_feats=tokens_to_feats(tokens.reshape(b,1,h,w))
                # ################## Generator ####################
                    after_quant_feat = net_g.vqgan.after_quant(quant_feats)
            
                    x=after_quant_feat
                    for j in range(net_g.max_depth):
                        cur_res = net_g.gt_res // 2**net_g.max_depth * 2**j
                        x = net_g.fuse_convs_dict[str(cur_res)](enc_feats[j].detach(), x, 1)
                        x = net_g.vqgan.decoder_group[j](x)
                        
                    o = net_g.vqgan.out_conv(x)

                    o = o[..., :h_old , :w_old ]
                    cv2.imwrite(os.path.join(parallel_folder, f'{imgname}_{i}.png'),tensor2img(o,rgb2bgr=True))
                    
                    o = tensor2img(o,rgb2bgr=False)
                
        pbar.update(1)
           
    pbar.close()


def visualize_masks_with_color(image, masks, rows, cols, gap_size=10):
    b, h, w = masks.shape

    # 将像素值为1的区域涂成黑色，像素值为0的区域涂成白色
    processed_masks = np.stack([masks*255]*3, axis=-1).astype(np.uint8)

    # 计算合并后的图像大小
    combined_h = rows * h + (rows - 1) * gap_size
    combined_w = cols * w + (cols - 1) * gap_size

    # 初始化合并后的图像，初始为白色
    combined_image = np.ones((combined_h, combined_w, 3), dtype=np.uint8) * 255

    # 将第一张掩码图替换为彩色图像
    if image is not None:
        combined_image[:h, :w, :] = image

    # 将处理后的掩码图按指定行列叠加，并留有间隔
    for i in range(1, rows * cols):
        y_start = (i // cols) * (h + gap_size)
        y_end = y_start + h
        x_start = (i % cols) * (w + gap_size)
        x_end = x_start + w
        combined_image[y_start:y_end, x_start:x_end, :] = processed_masks[i]

    return combined_image
if __name__ == '__main__':
    main()
