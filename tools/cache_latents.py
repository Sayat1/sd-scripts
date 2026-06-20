# latentsのdiskへの事前キャッシュを行う / cache latents to disk

import argparse
import math
from multiprocessing import Value
import os,sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from accelerate.utils import set_seed
import torch
from tqdm import tqdm

from library import config_util, flux_train_utils, flux_utils, strategy_base, strategy_flux, strategy_sd, strategy_sdxl, strategy_anima, anima_utils, anima_train_utils
import library.accelerator_setup as accelerator_setup
import library.args as args_util
import library.dataset as dataset_util
import library.model_io as model_io
from library import sdxl_train_util
from library import utils
import library.sai_model_spec as sai_model_spec
from library.config_util import (
    ConfigSanitizer,
    BlueprintGenerator,
)
from library.utils import setup_logging, add_logging_arguments

setup_logging()
import logging

logger = logging.getLogger(__name__)


def set_tokenize_strategy(is_sd: bool, is_sdxl: bool, is_flux: bool, args: argparse.Namespace, is_anima: bool = False) -> None:
    if is_flux:
        _, is_schnell, _ = flux_utils.check_flux_state_dict_diffusers_schnell(args.pretrained_model_name_or_path)
    else:
        is_schnell = False

    if is_sd:
        tokenize_strategy = strategy_sd.SdTokenizeStrategy(args.v2, args.max_token_length, args.tokenizer_cache_dir)
    elif is_sdxl:
        tokenize_strategy = strategy_sdxl.SdxlTokenizeStrategy(args.max_token_length, args.tokenizer_cache_dir)
    elif is_anima:
        tokenize_strategy = strategy_anima.AnimaTokenizeStrategy(
            qwen3_path=args.qwen3,
            t5_tokenizer_path=args.t5_tokenizer_path,
            qwen3_max_length=args.qwen3_max_token_length,
            t5_max_length=args.t5_max_token_length,
        )
    else:
        if args.t5xxl_max_token_length is None:
            if is_schnell:
                t5xxl_max_token_length = 256
            else:
                t5xxl_max_token_length = 512
        else:
            t5xxl_max_token_length = args.t5xxl_max_token_length

        logger.info(f"t5xxl_max_token_length: {t5xxl_max_token_length}")
        tokenize_strategy = strategy_flux.FluxTokenizeStrategy(t5xxl_max_token_length, args.tokenizer_cache_dir)
    strategy_base.TokenizeStrategy.set_strategy(tokenize_strategy)


def cache_to_disk(args: argparse.Namespace) -> None:
    setup_logging(args, reset=True)
    accelerator_setup.prepare_dataset_args(args, True)
    accelerator_setup.enable_high_vram(args)

    # assert args.cache_latents_to_disk, "cache_latents_to_disk must be True / cache_latents_to_diskはTrueである必要があります"
    args.cache_latents = True
    args.cache_latents_to_disk = True

    if args.cache_text_encoder_outputs:
        args.cache_text_encoder_outputs_to_disk = True

    use_dreambooth_method = args.in_json is None

    if args.seed is not None:
        set_seed(args.seed)  # 乱数系列を初期化する

    is_sd = not args.sdxl and not args.flux and not args.anima
    is_sdxl = args.sdxl
    is_flux = args.flux
    is_anima = args.anima

    if args.cache_text_encoder_outputs:
        assert (
            is_sdxl or is_flux or is_anima
        ), "Cache text encoder outputs to disk is only supported for SDXL, FLUX and Anima models / テキストエンコーダ出力のディスクキャッシュはSDXL, FLUX, 또는Anima에서만 유효합니다"

    set_tokenize_strategy(is_sd, is_sdxl, is_flux, args, is_anima)

    if is_sd or is_sdxl:
        latents_caching_strategy = strategy_sd.SdSdxlLatentsCachingStrategy(is_sd, True, args.vae_batch_size, args.skip_cache_check)
    elif is_anima:
        latents_caching_strategy = strategy_anima.AnimaLatentsCachingStrategy(True, args.vae_batch_size, args.skip_cache_check)
    else:
        latents_caching_strategy = strategy_flux.FluxLatentsCachingStrategy(True, args.vae_batch_size, args.skip_cache_check)
    strategy_base.LatentsCachingStrategy.set_strategy(latents_caching_strategy)

    # データセットを準備する
    use_user_config = args.dataset_config is not None
    if args.dataset_class is None:
        blueprint_generator = BlueprintGenerator(ConfigSanitizer(True, True, args.masked_loss, True))
        if use_user_config:
            logger.info(f"Loading dataset config from {args.dataset_config}")
            user_config = config_util.load_user_config(args.dataset_config)
            ignored = ["train_data_dir", "reg_data_dir", "in_json"]
            if any(getattr(args, attr) is not None for attr in ignored):
                logger.warning(
                    "ignoring the following options because config file is found: {0} / 設定ファイルが利用されるため以下のオプションは無視されます: {0}".format(
                        ", ".join(ignored)
                    )
                )
        else:
            if use_dreambooth_method:
                logger.info("Using DreamBooth method.")
                user_config = {
                    "datasets": [
                        {
                            "subsets": config_util.generate_dreambooth_subsets_config_by_subdirs(
                                args.train_data_dir, args.reg_data_dir
                            )
                        }
                    ]
                }
            else:
                logger.info("Training with captions.")
                user_config = {
                    "datasets": [
                        {
                            "subsets": [
                                {
                                    "image_dir": args.train_data_dir,
                                    "metadata_file": args.in_json,
                                }
                            ]
                        }
                    ]
                }

        blueprint = blueprint_generator.generate(user_config, args)
        train_dataset_group, val_dataset_group = config_util.generate_dataset_group_by_blueprint(blueprint.dataset_group)
    else:
        # use arbitrary dataset class
        train_dataset_group = dataset_util.load_arbitrary_dataset(args)
        val_dataset_group = None

    # acceleratorを準備する
    logger.info("prepare accelerator")
    args.deepspeed = False
    accelerator = accelerator_setup.prepare_accelerator(args)

    # mixed precisionに対応した型を用意しておき適宜castする
    weight_dtype, _ = accelerator_setup.prepare_dtype(args)
    vae_dtype = torch.float32 if args.no_half_vae else weight_dtype

    # 모델을 읽込む
    logger.info("load model")
    text_encoders = None
    if is_sd:
        _, vae, _, _ = model_io.load_target_model(args, weight_dtype, accelerator)
    elif is_sdxl:
        if args.cache_text_encoder_outputs:
            (_, text_encoder1, text_encoder2, vae, _, _, _) = sdxl_train_util.load_target_model(args, accelerator, "sdxl", weight_dtype)
            text_encoders = [text_encoder1, text_encoder2]
        else:
            (_, _, _, vae, _, _, _) = sdxl_train_util.load_target_model(args, accelerator, "sdxl", weight_dtype)
    elif is_anima:
        vae = qwen_image_autoencoder_kl.load_vae(
            args.vae,
            device="cpu",
            disable_mmap=True,
            spatial_chunk_size=args.vae_chunk_size,
            disable_cache=args.vae_disable_cache,
        )
        if args.cache_text_encoder_outputs:
            qwen3_text_encoder, _ = anima_utils.load_qwen3_text_encoder(args.qwen3, dtype=weight_dtype, device=accelerator.device)
            text_encoders = [qwen3_text_encoder]
    else:
        vae = flux_utils.load_ae(args.ae, weight_dtype, "cpu", disable_mmap=args.disable_mmap_load_safetensors)
        if args.cache_text_encoder_outputs:
            clip_l = flux_utils.load_clip_l(args.clip_l, weight_dtype, accelerator.device, disable_mmap=args.disable_mmap_load_safetensors)
            t5xxl = flux_utils.load_t5xxl(args.t5xxl, None, accelerator.device, disable_mmap=args.disable_mmap_load_safetensors)

            t5xxl_dtype = utils.str_to_dtype(args.t5xxl_dtype, weight_dtype)
            if t5xxl.dtype != t5xxl_dtype:
                if t5xxl.dtype == torch.float8_e4m3fn and t5xxl_dtype.itemsize() >= 2:
                    logger.warning(
                        "The loaded model is fp8, but the specified T5XXL dtype is larger than fp8.  This may cause a performance drop."
                        " / 로드된 모델은 fp8이지만, 지정된 T5XXL의 dtype가 fp8보다 고정밀입니다. 성능 저하가 발생할 수 있습니다."
                    )
                logger.info(f"Casting T5XXL model to {t5xxl_dtype}")
                t5xxl.to(t5xxl_dtype)
            text_encoders = [clip_l, t5xxl]


    if is_sd or is_sdxl:
        if torch.__version__ >= "2.0.0":  # PyTorch 2.0.0 以上対応のxformersなら以下が使える
            vae.set_use_memory_efficient_attention_xformers(args.xformers)

    vae.to(accelerator.device, dtype=vae_dtype)
    vae.requires_grad_(False)
    vae.eval()

    if text_encoders:
        for text_encoder in text_encoders:
            text_encoder.requires_grad_(False)
            text_encoder.eval()

        # build text encoder outputs caching strategy
        if is_sdxl:
            text_encoder_outputs_caching_strategy = strategy_sdxl.SdxlTextEncoderOutputsCachingStrategy(
                args.cache_text_encoder_outputs_to_disk, None, args.skip_cache_check, is_weighted=args.weighted_captions
            )
        elif is_flux:
            text_encoder_outputs_caching_strategy = strategy_flux.FluxTextEncoderOutputsCachingStrategy(
                args.cache_text_encoder_outputs_to_disk,
                args.text_encoder_batch_size,
                args.skip_cache_check,
                is_partial=False,
                apply_t5_attn_mask=args.apply_t5_attn_mask,
            )
        elif is_anima:
            text_encoder_outputs_caching_strategy = strategy_anima.AnimaTextEncoderOutputsCachingStrategy(
                args.cache_text_encoder_outputs_to_disk, args.text_encoder_batch_size, args.skip_cache_check, is_partial=False
            )
        strategy_base.TextEncoderOutputsCachingStrategy.set_strategy(text_encoder_outputs_caching_strategy)

        # build text encoding strategy
        if is_sdxl:
            text_encoding_strategy = strategy_sdxl.SdxlTextEncodingStrategy()
        elif is_flux:
            text_encoding_strategy = strategy_flux.FluxTextEncodingStrategy(args.apply_t5_attn_mask)
        elif is_anima:
            text_encoding_strategy = strategy_anima.AnimaTextEncodingStrategy()
        strategy_base.TextEncodingStrategy.set_strategy(text_encoding_strategy)

    # cache latents with dataset
    # TODO use DataLoader to speed up

    train_dataset_group.is_disk_cached_latents_is_expected

    train_dataset_group.new_cache_latents(vae, accelerator)

    # cache text encoder outputs
    if args.cache_text_encoder_outputs:
        train_dataset_group.new_cache_text_encoder_outputs(text_encoders, accelerator)

    accelerator.wait_for_everyone()
    accelerator.print(f"Finished caching latents to disk.")


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    add_logging_arguments(parser)
    args_util.add_sd_models_arguments(parser)
    sai_model_spec.add_model_spec_arguments(parser)
    args_util.add_training_arguments(parser, True)
    args_util.add_dataset_arguments(parser, True, True, True)
    args_util.add_masked_loss_arguments(parser)
    config_util.add_config_arguments(parser)
    train_util.add_dit_training_arguments(parser)
    #flux_train_utils.add_flux_train_arguments(parser)
    anima_train_utils.add_anima_training_arguments(parser)

    parser.add_argument("--sdxl", action="store_true", help="Use SDXL model / SDXLモデルを使用する")
    parser.add_argument("--flux", action="store_true", help="Use FLUX model / FLUXモデルを使用する")
    parser.add_argument("--anima", action="store_true", help="Use Anima model / Animaモデルを使用する")

    parser.add_argument(
        "--t5xxl_dtype",
        type=str,
        default=None,
        help="T5XXL model dtype, default: None (use mixed precision dtype) / T5XXLモデルのdtype, デフォルト: None (mixed precision의dtypeを使用)",
    )
    parser.add_argument(
        "--weighted_captions",
        action="store_true",
        default=False,
        help="Enable weighted captions in the standard style (token:1.3). No commas inside parens, or shuffle/dropout may break the decoder. / 「[token]」、「(token)」「(token:1.3)」のような重み付きキャプションを有効にする。カンマを括弧内に入れるとシャッフルやdropoutで重みづけがおかしくなるので注意",
    )

    parser.add_argument(
        "--no_half_vae",
        action="store_true",
        help="do not use fp16/bf16 VAE in mixed precision (use float VAE) / mixed precisionでも fp16/bf16 VAEを使わずfloat VAEを使う",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="[Deprecated] This option does not work. Existing .npz files are always checked. Use `--skip_cache_check` to skip the check."
        " / [非推奨] このオプションは機能しません。既存の .npz は常に検証されます。`--skip_cache_check` で検証をスキップできます。",
    )
    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args,_ = parser.parse_known_args()
    args = train_util.read_config_from_file(args, parser)

    cache_to_disk(args)
