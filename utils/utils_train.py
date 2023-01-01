from pathlib import Path
import os
import random
import numpy as np

import torch
import torch.nn.functional as F
import torchvision

from einops import rearrange

from utils import utils_html
from utils.utils import mean_pooling


def get_dataset(args, tokenizer):
    args.truncate_captions = True
    if args.dataset_keys is not None and args.dataset_keys != "":
        assert Path(args.dataset_keys).exists()
        with open(args.dataset_keys, 'r') as f:
            keys = [k.rstrip() for k in f.readlines()]
    else:
        keys = None
    if args.dataset == 'video_text':
        from mmvid_pytorch.loader import TextVideoDataset
        ds = TextVideoDataset(
            args.image_text_folder,
            text_len=args.text_seq_len,
            image_size=args.image_size,
            resize_ratio=args.resize_ratio,
            truncate_captions=args.truncate_captions,
            tokenizer=tokenizer,
            shuffle=args.is_shuffle,
            mode='video',
            deterministic=args.deterministic,
            frame_num=args.frame_num,
            frame_step=args.frame_step,
            return_vc=True,
            video_only=args.video_only,
            cache=args.dataset_cache,
            return_neg=args.negvc,
            drop_sentence=args.drop_sentence,
            keys=keys,
        )
    elif args.dataset == 'mp4_text':
        from mmvid_pytorch.loader import TextMP4Dataset
        ds = TextMP4Dataset(
            args.image_text_folder,
            text_len=args.text_seq_len,
            image_size=args.image_size,
            resize_ratio=args.resize_ratio,
            truncate_captions=args.truncate_captions,
            tokenizer=tokenizer,
            shuffle=args.is_shuffle,
            mode='video',
            deterministic=args.deterministic,
            frame_num=args.frame_num,
            frame_step=args.frame_step,
            return_vc=True,
            video_only=args.video_only,
            cache=args.dataset_cache,
            return_neg=args.negvc,
            drop_sentence=args.drop_sentence,
            keys=keys,
            new_method=args.new_method,
            use_emotion_only=args.use_emotion_only,
        )
    elif args.dataset == 'imagestack_text':
        from mmvid_pytorch.loader import TextImageStackDataset
        ds = TextImageStackDataset(
            args.image_text_folder,
            text_len=args.text_seq_len,
            image_size=args.image_size,
            resize_ratio=args.resize_ratio,
            truncate_captions=args.truncate_captions,
            tokenizer=tokenizer,
            shuffle=args.is_shuffle,
            mode='video',
            deterministic=args.deterministic,
            frame_num=args.frame_num,
            frame_step=args.frame_step,
            return_vc=True,
            cache=args.dataset_cache,
        )
    elif args.dataset == 'shape_attr':
        from mmvid_pytorch.loader_ext import ShapeAttrDataset
        ds = ShapeAttrDataset(
            args.image_text_folder,
            text_len=args.text_seq_len,
            image_size=args.image_size,
            resize_ratio=args.resize_ratio,
            truncate_captions=args.truncate_captions,
            tokenizer=tokenizer,
            shuffle=args.is_shuffle,
            mode='video',
            deterministic=args.deterministic,
            frame_num=args.frame_num,
            frame_step=args.frame_step,
            return_vc=True,
            cache=args.dataset_cache,
            attr_mode=args.attr_mode,
            keys=keys,
            return_neg=args.negvc,
        )
    elif args.dataset == 'vox':
        from mmvid_pytorch.loader_ext import VoxDataset
        ds = VoxDataset(
            args.image_text_folder,
            text_len=args.text_seq_len,
            image_size=args.image_size,
            resize_ratio=args.resize_ratio,
            truncate_captions=args.truncate_captions,
            tokenizer=tokenizer,
            shuffle=args.is_shuffle,
            mode='video',
            deterministic=args.deterministic,
            frame_num=args.frame_num,
            frame_step=args.frame_step,
            return_vc=True,
            video_only=args.video_only,
            cache=args.dataset_cache,
            return_neg=args.negvc,
            attr_mode=args.attr_mode,
        )
    elif args.dataset == 'iper':
        from mmvid_pytorch.loader_ext import iPERDataset
        ds = iPERDataset(
            args.image_text_folder,
            text_len=args.text_seq_len,
            image_size=args.image_size,
            resize_ratio=args.resize_ratio,
            truncate_captions=args.truncate_captions,
            tokenizer=tokenizer,
            shuffle=args.is_shuffle,
            mode='video',
            deterministic=args.deterministic,
            frame_num=args.frame_num,
            frame_step=args.frame_step,
            return_vc=True,
            video_only=args.video_only,
            cache=args.dataset_cache,
            return_neg=args.negvc,
            drop_sentence=args.drop_sentence,
            slow=args.slow,
            keys=keys,
        )
    else:
        raise NotImplementedError
    return ds


def get_vae_model(which_vae, vae_params=None, vae_path=None, image_size=None, args=None):
    # if vae_params is not NONE, RESUMING from custom DiscreteVAE(**vae_params)
    # weight loading is handled in dalle model
    if args is not None and args.dalle_path and (not args.finetune):
        vae_path = None

    if vae_path is not None:
        print("the vae path is: {}".format(vae_path))

    if which_vae == 'vqgan1024':
        from mmvid_pytorch.vae import VQGanVAE1024
        vae = VQGanVAE1024(vae_path=vae_path, image_size=image_size)
        vae_params = None
    else:
        # NOTE: See dalle_pytorch if you want to use OpenAI's VAE or custom VAE
        raise NotImplementedError
    return vae, vae_params


def get_optimizer(args, params):
    if args.optimizer == 'adam':
        from torch.optim import Adam
        opt = Adam(params, lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        from torch.optim import AdamW
        # vqgan uses betas (0.9, 0.95)
        opt = AdamW(params, lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    else:
        raise NotImplementedError
    return opt


def get_tokenizer(args):
    if args.which_tokenizer == 'simple':
        from mmvid_pytorch.tokenizer import SimpleTokenizer
        tokenizer = SimpleTokenizer()
    else:
        raise NotImplementedError
    return tokenizer


def get_fixed_language_model(args):
    tokenizer2, language_model = None, None
    if args.fixed_language_model == 'roberta-large':
        from transformers import RobertaTokenizer, RobertaModel
        tokenizer2 = RobertaTokenizer.from_pretrained('roberta-large')
        language_model = RobertaModel.from_pretrained('roberta-large').cuda()
        text_feature_dim = 1024

        @torch.no_grad()
        def encode_text(descriptions, device='cuda'):
            encoded_input = tokenizer2(
                descriptions,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=args.text_seq_len,
            )
            encoded_input = {
                'input_ids': encoded_input['input_ids'].to(device),
                'attention_mask': encoded_input['attention_mask'].to(device),
            }
            output = language_model(**encoded_input)
            embeddings = mean_pooling(output, encoded_input['attention_mask'])
            return embeddings
    else:
        raise NotImplementedError

    return tokenizer2, language_model, text_feature_dim, encode_text


def get_text_feature_extractor(args):
    text_feature_dim = 0
    text_feature_extractor = None
    if args.pretrained_text_feature == 'roberta':
        from transformers import RobertaTokenizer, RobertaModel
        text_feature_tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        text_feature_extractor = RobertaModel.from_pretrained('roberta-large').cuda()
        text_feature_dim = 1024

        @torch.no_grad()
        def encode_text(descriptions, device='cuda'):
            batch_size = len(descriptions)
            feature = []
            for b in range(batch_size):
                encoded_input = text_feature_tokenizer(descriptions[b], return_tensors='pt')
                encoded_input = {
                    'input_ids': encoded_input['input_ids'].to(device),
                    'attention_mask': encoded_input['attention_mask'].to(device),
                }
                output = text_feature_extractor(**encoded_input)
                feature.append(output.last_hidden_state.squeeze(0))
            return feature

    elif args.pretrained_text_feature == 'openai_clip':
        from tokenizer import SimpleTokenizer
        text_feature_extractor = torch.jit.load(
            args.openai_clip_model_path).cuda().eval()
        text_feature_tokenizer = SimpleTokenizer()
        text_feature_dim = 512
        context_length = text_feature_extractor.context_length
        dtype = text_feature_extractor.visual.conv1.weight.dtype

        @torch.no_grad()
        def encode_text(descriptions, device='cuda'):
            text_input = text_feature_tokenizer.tokenize(
                descriptions,
                context_length,
                truncate_text=True,
            ).squeeze(0).cuda()
            x = text_feature_extractor.token_embedding(text_input).type(dtype)  # [batch_size, n_ctx, d_model]
            x = x + text_feature_extractor.positional_embedding.type(dtype)
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = text_feature_extractor.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = text_feature_extractor.ln_final(x).type(dtype)
            return x.float()

    else:
        encode_text = lambda x: x

    return text_feature_dim, encode_text


def clip_encode_image(model, image):
    device = image.device
    input_resolution = model.input_resolution.item()
    if image.shape[2] != input_resolution:
        image = F.interpolate(image, (input_resolution, input_resolution))
    image_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(device)
    image_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(device)
    image_input = (image - image_mean[:, None, None]) / image_std[:, None, None]
    image_features = model.encode_image(image_input).float()
    image_features /= image_features.norm(dim=-1, keepdim=True)
    return image_features


def save_model(save_dir, params={}, states={}, name='dalle.pt'):
    path = save_dir / name  # string specifies which epoch or iter
    save_obj = {
        **params,
        **states,
    }
    os.makedirs(path.parent, exist_ok=True)
    torch.save(save_obj, path)


# lr scheduler
def dummy_lr_scheduler_step(*args):
    pass


def prepare_lr_scheduler(args, optimizer):
    if args.lr_scheduler == 'reducelronplateau':
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=2,
            cooldown=5,
            min_lr=1e-6,
            verbose=True,
        )

        def step(*args):
            scheduler.step(*args)

        return None, step

    elif args.lr_scheduler == 'steplr':
        from torch.optim.lr_scheduler import StepLR
        scheduler = StepLR(
            optimizer,
            step_size=args.lr_scheduler_step_size,
            gamma=0.5,
        )

        def step(*args):
            scheduler.step()

        return scheduler, step

    elif args.lr_scheduler == 'cosineannealinglr':
        from torch.optim.lr_scheduler import CosineAnnealingLR
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=args.lr_scheduler_step_size,
            eta_min=1e-6,
        )

        def step(*args):
            scheduler.step()

        return scheduler, step

    elif args.lr_scheduler == 'warmupdecaylr':
        from deepspeed.runtime.lr_schedules import WarmupDecayLR
        scheduler = WarmupDecayLR(
            optimizer,
            args.iters,
            1e-6,
            args.learning_rate,
            args.lr_scheduler_warmup,
        )

        def step(*args):
            scheduler.step()

        return scheduler, step

    elif args.lr_scheduler == 'warmuplr':
        from deepspeed.runtime.lr_schedules import WarmupLR
        scheduler = WarmupLR(
            optimizer,
            1e-6,
            args.learning_rate,
            args.lr_scheduler_warmup,
        )

        def step(*args):
            scheduler.step()

        return scheduler, step

    else:
        raise NotImplementedError


@torch.no_grad()
def visualize_train(args, dalle_module, tokenizer, data_batch, which_iter, webpage=None):
    text_description, text, frames, visuals = data_batch['description'], data_batch['text'], \
                                              data_batch['target'], data_batch['visual']
    if isinstance(visuals, (list, tuple)):
        visuals = torch.stack(visuals, dim=1)

    N_SAMPLE = min(args.n_sample, args.batch_size)  # 4
    N_PER_SAMPLE = args.n_per_sample  # 4
    N_FRAME = args.num_targets  # 8
    N_FRAME_ = args.num_targets + args.num_visuals * args.visual  # 8
    N_VISUAL = args.num_visuals
    IMAGE_SIZE = args.image_size
    LOG_SAMPLE_DIR = args.log_sample_dir
    which_cvae = 'vae' if args.use_cvae is None else 'cvae'

    generate_images = dalle_module.generate_images
    pnag_suffix = '_dynamic' if args.pnag_dynamic else ''
    blank_frame_nvc = torch.ones(N_PER_SAMPLE, N_VISUAL, 3, args.image_size, args.image_size).cuda()
    blank_frame_1 = torch.ones(1, 3, args.image_size, args.image_size).cuda()

    samples_img = []
    captions_img = []
    if args.use_html:
        samples_web = []
        captions_web = []
        nrow_web = []

    for j in range(N_SAMPLE):
        if args.fixed_language_model is None:
            # @Crane: add new method here
            if args.new_method:
                sample_text1 = text[0][j:j + 1]
                sample_text2 = text[1][j:j + 1]
                token_list1 = sample_text1.masked_select(sample_text1 != 0).tolist()
                token_list2 = sample_text2.masked_select(sample_text2 != 0).tolist()
                decoded_text1 = tokenizer.decode(token_list1)
                decoded_text2 = tokenizer.decode(token_list2)
                if isinstance(decoded_text1, (list, tuple)):
                    decoded_text1 = decoded_text1[0]
                if isinstance(decoded_text2, (list, tuple)):
                    decoded_text2 = decoded_text2[0]
                decoded_text = decoded_text1 + ". " + decoded_text2
                # containing two sentences
                text_repeat = [text[0][j:j + 1].repeat(N_PER_SAMPLE, 1),
                               text[1][j:j + 1].repeat(N_PER_SAMPLE, 1)]
            else:
                sample_text = text[j:j + 1]
                token_list = sample_text.masked_select(sample_text != 0).tolist()
                decoded_text = tokenizer.decode(token_list)
                if isinstance(decoded_text, (list, tuple)):
                    decoded_text = decoded_text[0]
                text_repeat = text[j:j + 1].repeat(N_PER_SAMPLE, 1)
        else:
            decoded_text = text_description[j]
            text_repeat = text[j:j + 1].repeat(N_PER_SAMPLE, 1)

        # Sample (with visual)
        face_mode = None
        frames_recon = dalle_module.recon_images(frames[j:j + 1, :N_FRAME])
        visual = visuals[j:j + 1, ...].repeat(N_PER_SAMPLE, 1, 1, 1, 1) if args.visual else None
        if args.visual:
            visual_real = visuals[j, ...]
            visual_recon = dalle_module.recon_images(visual_real, which_vae=which_cvae)
            samples_img.append(torch.cat((visual_real, frames[j, :N_FRAME, ...]), 0))  # real video sequence
            samples_img.append(torch.cat((visual_recon, frames_recon), 0))
            visual_prompt = visuals[j:j + 1, ...].clone().repeat(
                N_PER_SAMPLE, 1, 1, 1, 1)  # b n c h w
            if args.rand_visual:
                visual_prompt[:, :, :, IMAGE_SIZE // 2:, :] = 1
            if args.vc_mode == 'face_8x8':
                block_size = 16
                visual_prompt_ = torch.ones_like(visual_prompt)
                if random.random() < 0.5:
                    face_mode = 'eyes_nose'
                    visual_prompt_[:, :, :, 2 * block_size:5 * block_size,
                    1 * block_size:7 *
                                   block_size] = visual_prompt[:, :, :, 2 *
                                                                        block_size:5 *
                                                                                   block_size, 1 *
                                                                                               block_size:7 *
                                                                                                          block_size]
                else:
                    face_mode = 'mouth'
                    visual_prompt_[:, :, :, 5 * block_size:7 * block_size,
                    2 * block_size:6 *
                                   block_size] = visual_prompt[:, :, :, 5 *
                                                                        block_size:7 *
                                                                                   block_size, 2 *
                                                                                               block_size:6 *
                                                                                                          block_size]
                visual_prompt = visual_prompt_
            elif args.vc_mode == 'face2_8x8':
                block_size = 16
                visual_prompt_ = torch.ones_like(visual_prompt)
                visual_prompt_[:, 0, ...] = visual_prompt[:, 0, ...]
                face_mode = 'face2'
                visual_prompt_[:, 1:, :, 2 * block_size:6 * block_size,
                2 * block_size:6 *
                               block_size] = visual_prompt[:, 1:, :,
                                             2 * block_size:6 *
                                                            block_size,
                                             2 * block_size:6 *
                                                            block_size]
                visual_prompt = visual_prompt_
            elif args.vc_mode == 'mask2_8x8':
                block_size = 16
                visual_prompt_ = torch.ones_like(visual_prompt)
                visual_prompt_[:, :, :, 1 * block_size:7 * block_size,
                1 * block_size:7 *
                               block_size] = visual_prompt[:, :, :,
                                             1 * block_size:7 *
                                                            block_size,
                                             1 * block_size:7 *
                                                            block_size]
                visual_prompt = visual_prompt_
                face_mode = 'mask2'
            elif args.vc_mode == 'mask_8x8':
                block_size = 16
                visual_prompt_ = torch.ones_like(visual_prompt)
                visual_prompt_[:, :, :, 1 * block_size:7 * block_size,
                1 * block_size:7 *
                               block_size] = visual_prompt[:, :, :,
                                             1 * block_size:7 *
                                                            block_size,
                                             1 * block_size:7 *
                                                            block_size]
                visual_prompt = visual_prompt_
                face_mode = 'mask'
            elif args.vc_mode == 'shape_4x4':
                block_size = 16
                visual_prompt[:, :, :, 1 * block_size:3 * block_size,
                1 * block_size:3 * block_size] = 1
                face_mode = 'shape'
        else:
            samples_img.append(frames[j, :N_FRAME, ...])  # real video sequence
            samples_img.append(frames_recon)
        captions_img.append(f'{j + 1}. {decoded_text}')

        if args.use_html:
            nrow_web += [0]
            if args.visual:
                samples_web += list(torch.split(visual_real, 1, dim=0))
                samples_web += list(torch.split(visual_recon, 1, dim=0))
                captions_web += [f'vc_{jj + 1} [real]' for jj in range(N_VISUAL)]
                captions_web += [
                    f'vc_{jj + 1} [recon]' for jj in range(N_VISUAL)
                ]
                nrow_web[-1] += 2 * N_VISUAL
            samples_web.append(frames[j, :N_FRAME, ...])
            samples_web.append(frames_recon)
            captions_web += [decoded_text]
            captions_web += ['sequence [recon]']
            nrow_web[-1] += 2
        for mp_steps in args.mask_predict_steps:
            if mp_steps <= 0:
                mp_steps = args.mp_config['T']
            sample_vc, tmp, _ = generate_images(
                text_repeat,
                visual=visual,
                erase_visual=args.rand_visual,
                dynamic=args.pnag_dynamic,
                debug=args.debug,
                mask_predict_steps=mp_steps,
                vc_mode=args.vc_mode,
                face_mode=face_mode,
                mp_config=args.mp_config,
            )
            if args.visual:
                samples_img.append(torch.cat((visual_prompt, sample_vc), 1).reshape(N_PER_SAMPLE * N_FRAME_,
                                                                                    *frames.shape[2:5]))
            else:
                samples_img.append(sample_vc.reshape(N_PER_SAMPLE * N_FRAME, *frames.shape[2:5]))
            if args.use_html:
                nrow_web += [0]
                if args.visual:
                    samples_web += list(
                        torch.split(visual_prompt[0, ...], 1, dim=0))
                    captions_web += [
                        f'vc_{jj + 1} [prompt]' for jj in range(N_VISUAL)
                    ]
                    nrow_web[-1] += N_VISUAL
                samples_web += list(torch.split(sample_vc, 1, dim=0))
                captions_web += [
                    f'sample {jj + 1} [T={mp_steps}]'
                    for jj in range(N_PER_SAMPLE)
                ]
                nrow_web[-1] += N_PER_SAMPLE
            if args.debug:
                os.makedirs(LOG_SAMPLE_DIR / f'{which_iter}_pnag',
                            exist_ok=True)
                tmp.insert(0, frames[j, :N_FRAME, ...])
                tmp = torch.cat(tmp, 0)
                torchvision.utils.save_image(
                    tmp,
                    LOG_SAMPLE_DIR / f'{which_iter}_pnag' /
                    f'{j:02d}{pnag_suffix}_T={mp_steps}.png',
                    nrow=N_FRAME,
                    normalize=True,
                    range=(0, 1))
        mp_steps = args.mask_predict_steps1

        if args.visual:
            j2 = (j + 1) % frames.shape[0]
            visual_prompt = visuals[j2:j2 + 1, ...].clone().repeat(
                N_PER_SAMPLE, 1, 1, 1, 1)  # b n c h w
            if args.rand_visual:
                visual_prompt[:, :, :, IMAGE_SIZE // 2:, :] = 1
            if args.vc_mode == 'face_8x8':
                block_size = 16
                visual_prompt_ = torch.ones_like(visual_prompt)
                if random.random() < 0.5:
                    face_mode = 'eyes_nose'
                    visual_prompt_[:, :, :, 2 * block_size:5 * block_size,
                    1 * block_size:7 *
                                   block_size] = visual_prompt[:, :, :, 2 *
                                                                        block_size:5 *
                                                                                   block_size, 1 *
                                                                                               block_size:7 *
                                                                                                          block_size]
                else:
                    face_mode = 'mouth'
                    visual_prompt_[:, :, :, 5 * block_size:7 * block_size,
                    2 * block_size:6 *
                                   block_size] = visual_prompt[:, :, :, 5 *
                                                                        block_size:7 *
                                                                                   block_size, 2 *
                                                                                               block_size:6 *
                                                                                                          block_size]
                visual_prompt = visual_prompt_
                visual_cf = visuals[j2:j2 + 1, ...]
                visual_cf = visual_cf.repeat(N_PER_SAMPLE, 1, 1, 1, 1)
            elif args.vc_mode == 'face2_8x8':
                block_size = 16
                visual_prompt_ = torch.ones_like(visual_prompt)
                visual_prompt_[:, 0, ...] = visual_prompt[:, 0, ...]
                face_mode = 'face2'
                visual_prompt1 = visuals[j:j + 1,
                                 ...].repeat(N_PER_SAMPLE, 1, 1, 1, 1)
                visual_prompt_[:, 1:, :, 2 * block_size:6 * block_size,
                2 * block_size:6 *
                               block_size] = visual_prompt1[:, 1:, :,
                                             2 * block_size:6 *
                                                            block_size,
                                             2 * block_size:6 *
                                                            block_size]
                visual_prompt = visual_prompt_
                visual_cf = visuals[j:j + 1, ...].clone()  # !!!
                visual_cf[:, 0, ...] = visuals[j2:j2 + 1, 0, ...]
                visual_cf = visual_cf.repeat(N_PER_SAMPLE, 1, 1, 1, 1)
            elif args.vc_mode == 'mask2_8x8':
                block_size = 16
                visual_prompt_ = torch.ones_like(visual_prompt)
                visual_prompt1 = visuals[j:j + 1,
                                 ...].repeat(N_PER_SAMPLE, 1, 1, 1, 1)
                visual_prompt_[:, 0, :, 1 * block_size:7 * block_size,
                1 * block_size:7 *
                               block_size] = visual_prompt1[:, 0, :,
                                             1 * block_size:7 *
                                                            block_size,
                                             1 * block_size:7 *
                                                            block_size]
                visual_prompt_[:, 1, :, 1 * block_size:7 * block_size,
                1 * block_size:7 *
                               block_size] = visual_prompt[:, 1, :,
                                             1 * block_size:7 *
                                                            block_size,
                                             1 * block_size:7 *
                                                            block_size]
                visual_prompt = visual_prompt_
                visual_cf = visuals[j:j + 1, ...].clone()  # !!!
                visual_cf[:, 1, ...] = visuals[j2:j2 + 1, 1, ...]
                visual_cf = visual_cf.repeat(N_PER_SAMPLE, 1, 1, 1, 1)
                face_mode = 'mask2'
            elif args.vc_mode == 'mask_8x8':
                block_size = 16
                visual_prompt_ = torch.ones_like(visual_prompt)
                visual_prompt_[:, :, :, 1 * block_size:7 * block_size,
                1 * block_size:7 *
                               block_size] = visual_prompt[:, :, :,
                                             1 * block_size:7 *
                                                            block_size,
                                             1 * block_size:7 *
                                                            block_size]
                visual_prompt = visual_prompt_
                visual_cf = visuals[j2:j2 + 1,
                            ...].repeat(N_PER_SAMPLE, 1, 1, 1, 1)
                face_mode = 'mask'
            elif args.vc_mode == 'shape_4x4':
                block_size = 16
                visual_prompt[:, :, :, 1 * block_size:3 * block_size,
                1 * block_size:3 * block_size] = 1
                visual_cf = visuals[j2:j2 + 1, ...].repeat(N_PER_SAMPLE, 1, 1, 1, 1)
                face_mode = 'shape'
            else:
                visual_cf = visuals[j2:j2 + 1, ...].repeat(N_PER_SAMPLE, 1, 1, 1, 1)
            sample_cf, tmp, _ = generate_images(
                text_repeat,
                visual=visual_cf,
                erase_visual=args.rand_visual,
                dynamic=args.pnag_dynamic,
                debug=args.debug,
                mask_predict_steps=mp_steps,
                vc_mode=args.vc_mode,
                face_mode=face_mode,
                mp_config=args.mp_config,
            )
            samples_img.append(
                torch.cat((visual_prompt, sample_cf),
                          1).reshape(N_PER_SAMPLE * N_FRAME_,
                                     *frames.shape[2:5]))
            if args.use_html:
                samples_web += list(
                    torch.split(visual_prompt[0, ...], 1, dim=0)
                )
                samples_web += list(
                    torch.split(sample_cf, 1, dim=0)
                )
                captions_web += [
                    f'cf_{jj + 1} [prompt]' for jj in range(N_VISUAL)
                ]
                captions_web += [
                    f'sample {jj + 1}' for jj in range(N_PER_SAMPLE)
                ]
                nrow_web += [N_VISUAL + N_PER_SAMPLE]
            if args.debug:
                # tmp.insert(0, frames[j,:N_FRAME,...])
                tmp = torch.cat(tmp, 0)
                torchvision.utils.save_image(tmp,
                                             LOG_SAMPLE_DIR /
                                             f'{which_iter}_pnag' /
                                             f'cf_{j:02d}{pnag_suffix}.png',
                                             nrow=N_FRAME,
                                             normalize=True,
                                             range=(0, 1))

        if args.visual and not args.fullvc:
            sample_free, tmp, _ = generate_images(
                text_repeat,
                visual=None,
                erase_visual=args.rand_visual,
                dynamic=args.pnag_dynamic,
                debug=args.debug,
                mask_predict_steps=mp_steps,
                mp_config=args.mp_config,
            )
            samples_img.append(torch.cat((blank_frame_nvc, sample_free), 1).reshape(N_PER_SAMPLE * N_FRAME_,
                                                                                    *frames.shape[2:5]))
            if args.use_html:
                samples_web += [blank_frame_1] * N_VISUAL
                samples_web += list(torch.split(sample_free, 1, dim=0))
                captions_web += [f'null [prompt]' for jj in range(N_VISUAL)]
                captions_web += [
                    f'sample {jj + 1}' for jj in range(N_PER_SAMPLE)
                ]
                nrow_web += [N_VISUAL + N_PER_SAMPLE]
            if args.debug:
                tmp = torch.cat(tmp, 0)
                torchvision.utils.save_image(tmp,
                                             LOG_SAMPLE_DIR /
                                             f'{which_iter}_pnag' /
                                             f'free_{j:02d}{pnag_suffix}.png',
                                             nrow=N_FRAME,
                                             normalize=True,
                                             range=(0, 1))

    samples_img = torch.cat(samples_img)  # (448, 3, 128, 128)
    torchvision.utils.save_image(samples_img,
                                 LOG_SAMPLE_DIR / f'{which_iter}.png',
                                 nrow=N_FRAME_,
                                 normalize=True,
                                 range=(0, 1))

    with open(LOG_SAMPLE_DIR / f'{which_iter}.txt', 'w') as f:
        f.write('\n'.join(captions_img))

    if args.use_html:
        webpage.add_header(f'iteration {which_iter}')
        utils_html.save_grid(
            webpage=webpage,
            tensor=samples_web,
            caption=captions_web,
            name=which_iter,
            nrow=nrow_web,
            width=min(IMAGE_SIZE, 256),
        )


@torch.no_grad()
def visualize_test(args, dalle_module, tokenizer, data_batch, which_iter, webpage=None, description=None,
                   tokenizer2=None, language_model=None, **kwargs):
    text_description, text, frames, visuals = data_batch['description'], data_batch['text'], \
                                              data_batch['target'], data_batch['visual']
    visuals_neg = data_batch['visual_neg']
    erase_real = False

    num_sentences = 0
    if description is not None:  # NOTE: this is set via args.description
        erase_real = True
        bs = text.shape[0]
        description = [description] if not isinstance(description, list) else description
        if args.fixed_language_model is not None:
            encoded_input = tokenizer2(
                description,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=args.text_seq_len,
            )
            encoded_input = {
                'input_ids': encoded_input['input_ids'].cuda(),
                'attention_mask': encoded_input['attention_mask'].cuda(),
            }
            model_output = language_model(**encoded_input)
            text = mean_pooling(model_output, encoded_input['attention_mask'])
            text = text.repeat(bs, 1)
        else:
            if args.eval_mode != 'normal':
                if args.eval_mode == 'test_inter':
                    assert len(description) == 2, "invalid description number for test-time interpolation, only 2"
                text = []
                # split the input long sentence into two when applying test-time or phenaki-like operation
                for description_ in description:
                    text += [
                        tokenizer.tokenize(
                            description_,
                            args.text_seq_len,
                            truncate_text=True,
                        ).repeat(bs, 1).cuda()
                    ]
            else:  # normal mode
                tokenized_text = tokenizer.tokenize(
                    description[0],
                    args.text_seq_len,
                    truncate_text=True,
                )
                text = tokenized_text.repeat(bs, 1).cuda()

        text_description = description * bs if args.eval_mode == 'normal' else description

    if erase_real:
        frames.fill_(1)

    if isinstance(visuals, (list, tuple)):
        visuals = torch.stack(visuals, dim=1)

    N_SAMPLE = min(args.n_sample, args.batch_size)
    N_PER_SAMPLE = args.n_per_sample
    N_FRAME = args.num_targets
    N_FRAME_ = args.num_targets + args.num_visuals
    N_VISUAL = args.num_visuals
    IMAGE_SIZE = args.image_size
    LOG_SAMPLE_DIR = args.log_sample_dir
    which_cvae = 'vae' if args.use_cvae is None else 'cvae'

    # NOTE: this is previously hardcoded
    args.mask_predict_steps = [args.mp_config['T']]
    args.mask_predict_steps1 = args.mp_config['T']

    generate_images = dalle_module.generate_images
    pnag_suffix = '_dynamic' if args.pnag_dynamic else ''

    samples_img = []
    captions_img = []
    if args.use_html:
        samples_web = []
        captions_web = []
        nrow_web = []
    for j in range(N_SAMPLE):
        if args.fixed_language_model is None:
            if args.eval_mode != 'normal':  # test_inter & phenaki
                decoded_text = []
                text_repeat = []
                for text_element in text:
                    # text_element: [bs, text_seq_len]
                    sample_text = text_element[j:j + 1]  # [1, text_seq_len]
                    token_list = sample_text.masked_select(sample_text != 0).tolist()
                    decoded_text_ = tokenizer.decode(token_list)
                    if isinstance(decoded_text_, (list, tuple)):
                        decoded_text_ = decoded_text_[0]
                    decoded_text += [decoded_text_]
                    text_repeat += [text_element[j:j + 1].repeat(N_PER_SAMPLE, 1)]  # [num_sentences, n_per_sample, 70]

                # # old
                # sample_text1 = text[0][j:j + 1]
                # sample_text2 = text[1][j:j + 1]
                # token_list1 = sample_text1.masked_select(sample_text1 != 0).tolist()
                # token_list2 = sample_text2.masked_select(sample_text2 != 0).tolist()
                # decoded_text1 = tokenizer.decode(token_list1)
                # decoded_text2 = tokenizer.decode(token_list2)
                # if isinstance(decoded_text1, (list, tuple)):
                #     decoded_text1 = decoded_text1[0]
                # if isinstance(decoded_text2, (list, tuple)):
                #     decoded_text2 = decoded_text2[0]
                # decoded_text = decoded_text1 + ". " + decoded_text2
                # # containing two sentences
                # text_repeat = [text[0][j:j + 1].repeat(N_PER_SAMPLE, 1),
                #                text[1][j:j + 1].repeat(N_PER_SAMPLE, 1)]
            # elif args.eval_mode == "phenaki":  # phenaki
            #     decoded_text = []
            #     text_repeat = []
            #     for text_element in text:
            #         # text_element: [bs, text_seq_len]
            #         sample_text = text_element[j:j + 1]  # [1, text_seq_len]
            #         token_list = sample_text.masked_select(sample_text != 0).tolist()
            #         decoded_text_ = tokenizer.decode(token_list)
            #         if isinstance(decoded_text_, (list, tuple)):
            #             decoded_text_ = decoded_text_[0]
            #         decoded_text += [decoded_text_]
            #         text_repeat += [text_element[j:j + 1].repeat(N_PER_SAMPLE, 1)]  # [num_sentences, n_per_sample, 70]
            else:  # normal
                sample_text = text[j:j + 1]
                token_list = sample_text.masked_select(sample_text != 0).tolist()
                decoded_text = tokenizer.decode(token_list)
                if isinstance(decoded_text, (list, tuple)):
                    decoded_text = decoded_text[0]
                text_repeat = text[j:j + 1].repeat(N_PER_SAMPLE, 1)
        else:
            decoded_text = text_description[j]
            text_repeat = text[j:j + 1].repeat(N_PER_SAMPLE, 1)

        # Sample (with visual)
        face_mode = None
        frames_recon = dalle_module.recon_images(frames[j:j + 1, :N_FRAME, ...])
        visual = visuals[j:j + 1, ...].repeat(N_PER_SAMPLE, 1, 1, 1, 1) if args.visual else None
        if args.visual:
            visual_real = visuals[j, ...]
            visual_recon = dalle_module.recon_images(visual_real, which_vae=which_cvae)
            samples_img.append(
                torch.cat((visual_real, frames[j, :N_FRAME, ...]), 0))  # real video sequence
            samples_img.append(torch.cat((visual_recon, frames_recon), 0))
            visual_prompt = visuals[j:j + 1, ...].clone().repeat(N_PER_SAMPLE, 1, 1, 1, 1)  # b n c h w
            if args.rand_visual:
                visual_prompt[:, :, :, IMAGE_SIZE // 2:, :] = 1
            if args.vc_mode == 'face_8x8':
                block_size = 16
                visual_prompt_ = torch.ones_like(visual_prompt)
                if random.random() < 0.5:
                    face_mode = 'eyes_nose'
                    visual_prompt_[:, :, :, 2 * block_size:5 * block_size,
                    1 * block_size:7 *
                                   block_size] = visual_prompt[:, :, :, 2 *
                                                                        block_size:5 *
                                                                                   block_size, 1 *
                                                                                               block_size:7 *
                                                                                                          block_size]
                else:
                    face_mode = 'mouth'
                    visual_prompt_[:, :, :, 5 * block_size:7 * block_size,
                    2 * block_size:6 *
                                   block_size] = visual_prompt[:, :, :, 5 *
                                                                        block_size:7 *
                                                                                   block_size, 2 *
                                                                                               block_size:6 *
                                                                                                          block_size]
                visual_prompt = visual_prompt_
            elif args.vc_mode == 'face3_8x8':  # for mug evaluation
                block_size = 16
                visual_prompt_ = torch.ones_like(visual_prompt)
                face_mode = 'center'
                visual_prompt_[:, :, :, 2 * block_size:6 * block_size,
                2 * block_size:6 * block_size] = visual_prompt[:, :, :,
                                                 2 * block_size:6 * block_size,
                                                 2 * block_size:6 * block_size]
                visual_prompt = visual_prompt_
            elif args.vc_mode == 'face2_8x8':
                block_size = 16
                visual_prompt_ = torch.ones_like(visual_prompt)
                visual_prompt_[:, 0, ...] = visual_prompt[:, 0, ...]
                face_mode = 'face2'
                visual_prompt_[:, 1:, :, 2 * block_size:6 * block_size,
                2 * block_size:6 * block_size] = visual_prompt[:, 1:, :,
                                                 2 * block_size:6 * block_size,
                                                 2 * block_size:6 * block_size]
                visual_prompt = visual_prompt_
            elif args.vc_mode == 'mask2_8x8':
                block_size = 16
                visual_prompt_ = torch.ones_like(visual_prompt)
                visual_prompt_[:, :, :, 1 * block_size:7 * block_size,
                1 * block_size:7 * block_size] = visual_prompt[:, :, :,
                                                 1 * block_size:7 * block_size,
                                                 1 * block_size:7 * block_size]
                visual_prompt = visual_prompt_
                face_mode = 'mask2'
            elif args.vc_mode == 'mask_8x8':
                block_size = 16
                visual_prompt_ = torch.ones_like(visual_prompt)
                visual_prompt_[:, :, :, 1 * block_size:7 * block_size,
                1 * block_size:7 * block_size] = visual_prompt[:, :, :,
                                                 1 * block_size:7 * block_size,
                                                 1 * block_size:7 * block_size]
                visual_prompt = visual_prompt_
                face_mode = 'mask'
            elif args.vc_mode == 'shape_4x4':
                block_size = 16
                visual_prompt[:, :, :, 1 * block_size:3 * block_size, 1 * block_size:3 * block_size] = 1
                face_mode = 'shape'

        if isinstance(decoded_text, list):
            for decoded_text_ in decoded_text:
                captions_img.append(f'{decoded_text_} ')
        else:
            captions_img.append(f'{j + 1}. {decoded_text}')
        if args.use_html:
            nrow_web += [0]
            if args.visual:
                samples_web += list(torch.split(visual_real, 1, dim=0))
                samples_web += list(torch.split(visual_recon, 1, dim=0))
                captions_web += [f'vc_{jj + 1} [real]' for jj in range(N_VISUAL)]
                captions_web += [f'vc_{jj + 1} [recon]' for jj in range(N_VISUAL)]
                nrow_web[-1] += 2 * N_VISUAL
            samples_web.append(frames[j, :N_FRAME, ...])
            samples_web.append(frames_recon)
            captions_web += [decoded_text]
            captions_web += ['sequence [recon]']
            nrow_web[-1] += 2

        # conditional sampling following Phenaki
        if args.eval_mode == "phenaki":
            for mp_steps in args.mask_predict_steps:
                if mp_steps <= 0:
                    mp_steps = args.mp_config['T']
                print(f"mp_steps is {mp_steps}")
                sample_vc = []
                code_prev = None
                for t in range(num_sentences):
                    # sample images
                    sample_vc_, _, code_ = generate_images(
                        text_repeat[t],
                        visual=visual,
                        erase_visual=args.rand_visual,
                        dynamic=args.pnag_dynamic,
                        debug=args.debug,
                        mask_predict_steps=mp_steps,
                        mp_config=args.mp_config,
                        preserve=code_prev,
                        pc_mode=args.pc_mode,
                        t_overlap=0 if t == 0 else args.t_overlap,
                    )
                    code_prev = code_
                    # if t == 0:
                    #     sample_vc.append(sample_vc_)
                    # else:
                    #     sample_vc.append(sample_vc_[:, t_overlap:, ...])
                    sample_vc.append(sample_vc_)
                sample_vc = torch.cat(sample_vc, dim=1)

                if args.use_html:
                    nrow_web += [0]
                    samples_web += list(torch.split(sample_vc, 1, dim=0))
                    captions_web += [
                        f'sample {jj + 1} [T={mp_steps}]' for jj in range(N_PER_SAMPLE)
                    ]
                    nrow_web[-1] += N_PER_SAMPLE
                samples_img.append(sample_vc.reshape(-1, *frames.shape[2:5]))

        else:
            for mp_steps in args.mask_predict_steps:
                if mp_steps <= 0:
                    mp_steps = args.mp_config['T']
                # sample images
                sample_vc, tmp, _ = generate_images(
                    text_repeat,
                    visual=visual,
                    erase_visual=args.rand_visual,
                    dynamic=args.pnag_dynamic,
                    debug=args.debug,
                    mask_predict_steps=mp_steps,
                    mp_config=args.mp_config,
                    face_mode=face_mode,
                )
                if args.visual:
                    samples_img.append(torch.cat((visual_prompt, sample_vc), 1).reshape(N_PER_SAMPLE * N_FRAME_,
                                                                                        *frames.shape[2:5]))
                else:
                    samples_img.append(sample_vc.reshape(N_PER_SAMPLE * N_FRAME, *frames.shape[2:5]))
                if args.use_html:
                    nrow_web += [0]
                    if args.visual:
                        samples_web += list(torch.split(visual_prompt[0, ...], 1, dim=0))
                        captions_web += [f'vc_{jj + 1} [prompt]' for jj in range(N_VISUAL)]
                        nrow_web[-1] += N_VISUAL
                    samples_web += list(torch.split(sample_vc, 1, dim=0))
                    captions_web += [f'sample {jj + 1} [T={mp_steps}]' for jj in range(N_PER_SAMPLE)]
                    nrow_web[-1] += N_PER_SAMPLE
                if args.debug:
                    os.makedirs(LOG_SAMPLE_DIR / f'{which_iter}_pnag', exist_ok=True)
                    tmp.insert(0, frames[j, :N_FRAME, ...])
                    tmp = torch.cat(tmp, 0)
                    torchvision.utils.save_image(
                        tmp,
                        LOG_SAMPLE_DIR / f'{which_iter}_pnag' /
                        f'{j:02d}{pnag_suffix}_T={mp_steps}.png',
                        nrow=N_FRAME,
                        normalize=True,
                        range=(0, 1))

        mp_steps = args.mask_predict_steps1

        if args.visual and args.test_mode is None:
            j2 = (j + 1) % frames.shape[0]
            visual_prompt = visuals[j2:j2 + 1, ...].clone().repeat(
                N_PER_SAMPLE, 1, 1, 1, 1)  # b n c h w
            if args.rand_visual:
                visual_prompt[:, :, :, IMAGE_SIZE // 2:, :] = 1
            if args.vc_mode == 'face_8x8':
                block_size = 16
                visual_prompt_ = torch.ones_like(visual_prompt)
                if random.random() < 0.5:
                    face_mode = 'eyes_nose'
                    visual_prompt_[:, :, :, 2 * block_size:5 * block_size,
                    1 * block_size:7 *
                                   block_size] = visual_prompt[:, :, :, 2 * block_size:5 * block_size,
                                                 1 * block_size:7 * block_size]
                else:
                    face_mode = 'mouth'
                    visual_prompt_[:, :, :, 5 * block_size:7 * block_size,
                    2 * block_size:6 *
                                   block_size] = visual_prompt[:, :, :, 5 *
                                                                        block_size:7 *
                                                                                   block_size, 2 *
                                                                                               block_size:6 *
                                                                                                          block_size]
                visual_prompt = visual_prompt_
                visual_cf = visuals[j2:j2 + 1, ...]
                visual_cf = visual_cf.repeat(N_PER_SAMPLE, 1, 1, 1, 1)
            elif args.vc_mode == 'face2_8x8':
                block_size = 16
                visual_prompt_ = torch.ones_like(visual_prompt)
                visual_prompt_[:, 0, ...] = visual_prompt[:, 0, ...]
                face_mode = 'face2'
                visual_prompt1 = visuals[j:j + 1, ...].repeat(N_PER_SAMPLE, 1, 1, 1, 1)
                visual_prompt_[:, 1:, :, 2 * block_size:6 * block_size,
                2 * block_size:6 *
                               block_size] = visual_prompt1[:, 1:, :,
                                             2 * block_size:6 * block_size,
                                             2 * block_size:6 * block_size]
                visual_prompt = visual_prompt_
                visual_cf = visuals[j:j + 1, ...].clone()  # !!!
                visual_cf[:, 0, ...] = visuals[j2:j2 + 1, 0, ...]
                visual_cf = visual_cf.repeat(N_PER_SAMPLE, 1, 1, 1, 1)
            elif args.vc_mode == 'mask2_8x8':
                block_size = 16
                visual_prompt_ = torch.ones_like(visual_prompt)
                visual_prompt1 = visuals[j:j + 1, ...].repeat(N_PER_SAMPLE, 1, 1, 1, 1)
                visual_prompt_[:, 0, :, 1 * block_size:7 * block_size,
                1 * block_size:7 *
                               block_size] = visual_prompt1[:, 0, :,
                                             1 * block_size:7 * block_size,
                                             1 * block_size:7 * block_size]
                visual_prompt_[:, 1, :, 1 * block_size:7 * block_size,
                1 * block_size:7 *
                               block_size] = visual_prompt[:, 1, :,
                                             1 * block_size:7 * block_size,
                                             1 * block_size:7 * block_size]
                visual_prompt = visual_prompt_
                visual_cf = visuals[j:j + 1, ...].clone()
                visual_cf[:, 1, ...] = visuals[j2:j2 + 1, 1, ...]
                visual_cf = visual_cf.repeat(N_PER_SAMPLE, 1, 1, 1, 1)
                face_mode = 'mask2'
            elif args.vc_mode == 'mask_8x8':
                block_size = 16
                visual_prompt_ = torch.ones_like(visual_prompt)
                visual_prompt_[:, :, :, 1 * block_size:7 * block_size,
                1 * block_size:7 *
                               block_size] = visual_prompt[:, :, :,
                                             1 * block_size:7 *
                                                            block_size,
                                             1 * block_size:7 *
                                                            block_size]
                visual_prompt = visual_prompt_
                visual_cf = visuals[j2:j2 + 1, ...].repeat(N_PER_SAMPLE, 1, 1, 1, 1)
                face_mode = 'mask'
            elif args.vc_mode == 'shape_4x4':
                block_size = 16
                visual_prompt[:, :, :, 1 * block_size:3 * block_size,
                1 * block_size:3 * block_size] = 1
                visual_cf = visuals[j2:j2 + 1, ...].repeat(N_PER_SAMPLE, 1, 1, 1, 1)
                face_mode = 'shape'
            else:
                visual_cf = visuals[j2:j2 + 1, ...].repeat(N_PER_SAMPLE, 1, 1, 1, 1)
            sample_cf, tmp, _ = generate_images(
                text_repeat,
                visual=visual_cf,
                erase_visual=args.rand_visual,
                dynamic=args.pnag_dynamic,
                debug=args.debug,
                mask_predict_steps=mp_steps,
                mp_config=args.mp_config,
                face_mode=face_mode,
            )
            samples_img.append(
                torch.cat((visual_prompt, sample_cf), dim=1).reshape(N_PER_SAMPLE * N_FRAME_, *frames.shape[2:5])
            )
            if args.use_html:
                samples_web += list(torch.split(visual_prompt[0, ...], 1, dim=0))
                samples_web += list(torch.split(sample_cf, 1, dim=0))
                captions_web += [f'cf_{jj + 1} [prompt]' for jj in range(N_VISUAL)]
                captions_web += [f'sample {jj + 1}' for jj in range(N_PER_SAMPLE)]
                nrow_web += [N_VISUAL + N_PER_SAMPLE]
            if args.debug:
                tmp = torch.cat(tmp, 0)
                torchvision.utils.save_image(tmp,
                                             LOG_SAMPLE_DIR /
                                             f'{which_iter}_pnag' /
                                             f'cf_{j:02d}{pnag_suffix}.png',
                                             nrow=N_FRAME,
                                             normalize=True,
                                             range=(0, 1))

        # test_mode: for shapes
        # =================== for shapes ======================
        if args.visual and args.test_mode == 'shapes':
            for kk in range(3):
                visual_prompt = visuals[j:j + 1, ...].clone()
                visual_prompt[:, kk, ...] = visuals_neg[j:j + 1, kk, ...]
                visual_prompt = visual_prompt.repeat(N_PER_SAMPLE, 1, 1, 1, 1)
                sample_cf, tmp, _ = generate_images(
                    text_repeat,
                    visual=visual_prompt,
                    erase_visual=args.rand_visual,
                    # argmax=args.pnag_argmax,
                    dynamic=args.pnag_dynamic,
                    debug=args.debug,
                    mask_predict_steps=mp_steps,
                    mp_config=args.mp_config,
                )
                if args.rand_visual:
                    visual_prompt[:, :, :, IMAGE_SIZE // 2:, :] = 1
                samples_img.append(
                    torch.cat((visual_prompt, sample_cf),
                              1).reshape(N_PER_SAMPLE * N_FRAME_,
                                         *frames.shape[2:5]))
                if args.use_html:
                    samples_web += list(
                        torch.split(visual_prompt[0, ...], 1, dim=0))
                    samples_web += list(torch.split(sample_cf, 1, dim=0))
                    captions_web += [
                        f'cf-{kk + 1}_{jj + 1} [prompt]' for jj in range(N_VISUAL)
                    ]
                    captions_web += [
                        f'sample {jj + 1} [T={mp_steps}]'
                        for jj in range(N_PER_SAMPLE)
                    ]
                    nrow_web += [N_VISUAL + N_PER_SAMPLE]
        # ========================================================

    samples_img = torch.cat(samples_img)
    torchvision.utils.save_image(samples_img,
                                 LOG_SAMPLE_DIR / f'{which_iter}.png',
                                 nrow=N_FRAME_ + args.num_targets * (num_sentences - 1)
                                 if args.eval_mode == 'phenaki' else N_FRAME_,
                                 normalize=True,
                                 range=(0, 1))

    with open(LOG_SAMPLE_DIR / f'{which_iter}.txt', 'w') as f:
        f.write('\n'.join(captions_img))

    if args.use_html:
        webpage.add_header(f'iteration {which_iter}')
        utils_html.save_grid(
            webpage=webpage,
            tensor=samples_web,
            caption=captions_web,
            name=which_iter,
            nrow=nrow_web,
            width=min(IMAGE_SIZE, 256),
        )
