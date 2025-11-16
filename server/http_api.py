import argparse
import asyncio
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional
import uuid

import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

import wan
from wan.configs import SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS
from wan.utils.multitalk_utils import save_video_ffmpeg
from wan.utils.utils import str2bool

from generate_infinitetalk import (
    _validate_args as validate_generation_args,
    audio_prepare_multi,
    audio_prepare_single,
    custom_init,
    get_embedding,
)


logger = logging.getLogger("infinitetalk.http")


def _parse_args():
    parser = argparse.ArgumentParser(description="InfiniteTalk FastAPI 服务")
    parser.add_argument(
        "--task",
        type=str,
        default="infinitetalk-14B",
        choices=list(WAN_CONFIGS.keys()),
        help="要加载的任务配置。")
    parser.add_argument(
        "--size",
        type=str,
        default="infinitetalk-480",
        choices=list(SIZE_CONFIGS.keys()),
        help="生成分辨率 bucket。")
    parser.add_argument(
        "--mode",
        type=str,
        default="streaming",
        choices=["clip", "streaming"],
        help="推理模式 clip/streaming。")
    parser.add_argument(
        "--frame_num",
        type=int,
        default=81,
        help="每个片段生成帧数 (4n+1)。")
    parser.add_argument(
        "--max_frame_num",
        type=int,
        default=1000,
        help="streaming 时的最长帧数。")
    parser.add_argument(
        "--motion_frame",
        type=int,
        default=9,
        help="长视频驱动帧长度。")
    parser.add_argument(
        "--sample_steps",
        type=int,
        default=None,
        help="默认采样步数，空值按模型自动设置。")
    parser.add_argument(
        "--sample_shift",
        type=float,
        default=None,
        help="flow matching 调度器偏移。")
    parser.add_argument(
        "--sample_text_guide_scale",
        type=float,
        default=5.0,
        help="文本 CFG。")
    parser.add_argument(
        "--sample_audio_guide_scale",
        type=float,
        default=4.0,
        help="音频 CFG。")
    parser.add_argument(
        "--base_seed",
        type=int,
        default=42,
        help="推理种子。")
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default="weights/Wan2.1-I2V-14B-480P",
        help="Wan 主模型路径。")
    parser.add_argument(
        "--wav2vec_dir",
        type=str,
        default="weights/chinese-wav2vec2-base",
        help="wav2vec2 权重路径。")
    parser.add_argument(
        "--infinitetalk_dir",
        type=str,
        default="weights/InfiniteTalk/single/infinitetalk.safetensors",
        help="InfiniteTalk LoRA 权重路径。")
    parser.add_argument(
        "--quant_dir",
        type=str,
        default=None,
        help="量化模型路径。")
    parser.add_argument(
        "--quant",
        type=str,
        default=None,
        help="量化类型 (如 int8/fp8)。")
    parser.add_argument(
        "--dit_path",
        type=str,
        default=None,
        help="自定义 DiT 权重。")
    parser.add_argument(
        "--lora_dir",
        type=str,
        nargs="*",
        default=None,
        help="额外 LoRA 文件路径列表。")
    parser.add_argument(
        "--lora_scale",
        type=float,
        nargs="*",
        default=[1.2],
        help="对应 LoRA 的缩放。")
    parser.add_argument(
        "--ulysses_size",
        type=int,
        default=1,
        help="Ulysses 并行度。")
    parser.add_argument(
        "--ring_size",
        type=int,
        default=1,
        help="Ring Attention 并行度。")
    parser.add_argument(
        "--t5_fsdp",
        action="store_true",
        default=False,
        help="是否对 T5 启用 FSDP。")
    parser.add_argument(
        "--dit_fsdp",
        action="store_true",
        default=False,
        help="是否对 DiT 启用 FSDP。")
    parser.add_argument(
        "--t5_cpu",
        action="store_true",
        default=False,
        help="将 T5 放在 CPU。")
    parser.add_argument(
        "--offload_model",
        type=str2bool,
        default=None,
        help="推理后是否将权重卸载到 CPU。")
    parser.add_argument(
        "--num_persistent_param_in_dit",
        type=int,
        default=None,
        help="DiT VRAM 管理参数。")
    parser.add_argument(
        "--audio_save_dir",
        type=str,
        default="save_audio/http_api",
        help="音频缓存目录。")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="runs/http_api",
        help="生成视频输出目录。")
    parser.add_argument(
        "--color_correction_strength",
        type=float,
        default=1.0,
        help="颜色校正强度。")
    parser.add_argument(
        "--use_teacache",
        action="store_true",
        default=False,
        help="启用 TeaCache。")
    parser.add_argument(
        "--teacache_thresh",
        type=float,
        default=0.2,
        help="TeaCache 阈值。")
    parser.add_argument(
        "--use_apg",
        action="store_true",
        default=False,
        help="启用 APG。")
    parser.add_argument(
        "--apg_momentum",
        type=float,
        default=-0.75,
        help="APG momentum。")
    parser.add_argument(
        "--apg_norm_threshold",
        type=float,
        default=55,
        help="APG norm 阈值。")
    parser.add_argument(
        "--server_host",
        type=str,
        default="0.0.0.0",
        help="HTTP 服务监听地址。")
    parser.add_argument(
        "--server_port",
        type=int,
        default=8888,
        help="HTTP 服务端口。")
    parser.add_argument(
        "--allow_origins",
        type=str,
        nargs="*",
        default=["*"],
        help="CORS 允许的来源。")
    args = parser.parse_args()
    validate_generation_args(args)
    return args


def _init_models(args):
    if args.offload_model is None:
        args.offload_model = True

    cfg = WAN_CONFIGS[args.task]
    if args.ulysses_size > 1:
        assert cfg.num_heads % args.ulysses_size == 0, "`ulysses_size` 必须整除 num_heads。"

    pipeline = wan.InfiniteTalkPipeline(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        quant_dir=args.quant_dir,
        device_id=0,
        rank=0,
        t5_fsdp=args.t5_fsdp,
        dit_fsdp=args.dit_fsdp,
        use_usp=(args.ulysses_size > 1 or args.ring_size > 1),
        t5_cpu=args.t5_cpu,
        lora_dir=args.lora_dir,
        lora_scales=args.lora_scale,
        quant=args.quant,
        dit_path=args.dit_path,
        infinitetalk_dir=args.infinitetalk_dir,
    )
    if args.num_persistent_param_in_dit is not None:
        pipeline.vram_management = True
        pipeline.enable_vram_management(
            num_persistent_param_in_dit=args.num_persistent_param_in_dit)

    wav2vec_feature_extractor, audio_encoder = custom_init('cpu',
                                                           args.wav2vec_dir)
    return pipeline, wav2vec_feature_extractor, audio_encoder


def _ensure_dir(path_str: str) -> Path:
    path = Path(path_str)
    path.mkdir(parents=True, exist_ok=True)
    return path


async def _persist_upload(file: UploadFile, target_path: Path):
    data = await file.read()
    target_path.write_bytes(data)
    await file.close()
    return target_path


def _prepare_single_audio(audio_path: Path, cache_dir: Path,
                          wav2vec_feature_extractor, audio_encoder):
    speech = audio_prepare_single(str(audio_path))
    audio_embedding = get_embedding(speech, wav2vec_feature_extractor,
                                    audio_encoder)
    emb_path = cache_dir / "audio_embedding.pt"
    sum_audio_path = cache_dir / "sum.wav"
    sf.write(sum_audio_path, speech, 16000)
    torch.save(audio_embedding, emb_path)
    return emb_path, sum_audio_path


def _prepare_dual_audio(left_path: Path, right_path: Path, cache_dir: Path,
                        audio_type: str, wav2vec_feature_extractor,
                        audio_encoder):
    new_left, new_right, sum_audio = audio_prepare_multi(
        str(left_path), str(right_path), audio_type)
    left_emb = get_embedding(new_left, wav2vec_feature_extractor,
                             audio_encoder)
    right_emb = get_embedding(new_right, wav2vec_feature_extractor,
                              audio_encoder)
    sum_audio_path = cache_dir / "sum.wav"
    sf.write(sum_audio_path, sum_audio, 16000)
    left_emb_path = cache_dir / "person1.pt"
    right_emb_path = cache_dir / "person2.pt"
    torch.save(left_emb, left_emb_path)
    torch.save(right_emb, right_emb_path)
    return left_emb_path, right_emb_path, sum_audio_path


def create_app(args):
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
    )

    pipeline, wav2vec_feature_extractor, audio_encoder = _init_models(args)
    audio_cache_dir = _ensure_dir(args.audio_save_dir)
    output_dir = _ensure_dir(args.output_dir)

    app = FastAPI(title="InfiniteTalk HTTP API", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=args.allow_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.pipeline = pipeline
    app.state.wav2vec_feature_extractor = wav2vec_feature_extractor
    app.state.audio_encoder = audio_encoder
    app.state.args = args
    app.state.lock = asyncio.Lock()
    app.state.audio_cache_dir = audio_cache_dir
    app.state.output_dir = output_dir

    @app.get("/healthz")
    async def health_check():
        return {"status": "ok"}

    @app.get("/files/{run_id}/{filename}")
    async def download_file(run_id: str, filename: str):
        file_path = app.state.output_dir / run_id / filename
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="文件不存在")
        return FileResponse(file_path)

    @app.post("/generate")
    async def generate_video(
        prompt: str = Form(...),
        inference_mode: str = Form(None),
        input_type: str = Form("image"),
        audio_type: str = Form("para"),
        sample_steps: Optional[int] = Form(None),
        text_guide_scale: Optional[float] = Form(None),
        audio_guide_scale: Optional[float] = Form(None),
        frame_num: Optional[int] = Form(None),
        motion_frame: Optional[int] = Form(None),
        media: UploadFile = File(...),
        audio: UploadFile = File(...),
        audio_secondary: UploadFile = File(None),
    ):
        args = app.state.args
        mode = inference_mode or args.mode
        if mode not in ("clip", "streaming"):
            raise HTTPException(status_code=400, detail="mode 仅支持 clip/streaming")

        run_stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        run_id = f"{run_stamp}_{uuid.uuid4().hex[:8]}"
        run_dir = app.state.output_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        media_suffix = Path(media.filename or "condition").suffix or ".bin"
        media_path = run_dir / f"condition{media_suffix}"
        await _persist_upload(media, media_path)

        audio_name = Path(audio.filename or "audio.wav")
        primary_audio_path = run_dir / f"audio{(audio_name.suffix or '.wav')}"
        await _persist_upload(audio, primary_audio_path)

        cond_audio = {}
        sum_audio_path = None

        if audio_secondary is not None and audio_secondary.filename:
            secondary_name = Path(audio_secondary.filename)
            secondary_path = run_dir / f"audio_right{(secondary_name.suffix or '.wav')}"
            await _persist_upload(audio_secondary, secondary_path)
            left_emb, right_emb, sum_audio_path = _prepare_dual_audio(
                primary_audio_path,
                secondary_path,
                run_dir,
                audio_type,
                app.state.wav2vec_feature_extractor,
                app.state.audio_encoder,
            )
            cond_audio["person1"] = str(left_emb)
            cond_audio["person2"] = str(right_emb)
        else:
            emb_path, sum_audio_path = _prepare_single_audio(
                primary_audio_path,
                run_dir,
                app.state.wav2vec_feature_extractor,
                app.state.audio_encoder,
            )
            cond_audio["person1"] = str(emb_path)

        input_clip = {
            "prompt": prompt,
            "cond_video": str(media_path),
            "cond_audio": cond_audio,
            "video_audio": str(sum_audio_path),
        }
        if input_type:
            input_clip["input_type"] = input_type

        inference_kwargs = dict(
            size_buckget=args.size,
            motion_frame=motion_frame or args.motion_frame,
            frame_num=frame_num or args.frame_num,
            shift=args.sample_shift,
            sampling_steps=sample_steps or args.sample_steps,
            text_guide_scale=text_guide_scale or args.sample_text_guide_scale,
            audio_guide_scale=audio_guide_scale or args.sample_audio_guide_scale,
            seed=args.base_seed,
            offload_model=args.offload_model,
            max_frames_num=(frame_num or args.frame_num) if mode == "clip" else args.max_frame_num,
            color_correction_strength=args.color_correction_strength,
            extra_args=args,
        )

        logger.info("开始生成: run_id=%s prompt=\"%s\"", run_id, prompt[:80])
        async with app.state.lock:
            try:
                video_tensor = app.state.pipeline.generate_infinitetalk(
                    input_clip,
                    **inference_kwargs,
                )
            except Exception as err:  # noqa: BLE001
                logger.exception("推理失败 run_id=%s", run_id)
                raise HTTPException(status_code=500, detail=str(err)) from err

        output_stub = run_dir / "result"
        save_video_ffmpeg(
            video_tensor,
            str(output_stub),
            [str(sum_audio_path)],
            high_quality_save=False,
        )
        mp4_path = output_stub.with_suffix(".mp4")

        response = {
            "status": "completed",
            "run_id": run_id,
            "video_path": str(mp4_path),
            "download_url": f"/files/{run_id}/{mp4_path.name}",
        }
        logger.info("生成完成 run_id=%s 输出=%s", run_id, mp4_path)
        return JSONResponse(response)

    return app


def main():
    args = _parse_args()
    app = create_app(args)
    uvicorn.run(app, host=args.server_host, port=args.server_port)


if __name__ == "__main__":
    main()
