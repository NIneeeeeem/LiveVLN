MODEL_PATH=./cache/Qwen2.5-VL_3B_sft_r2r_rxr_and_dagger_double_with_idm_and_scalevln_and_envdrop_traj_summary

CUDA_VISIBLE_DEVICES=0 vllm serve $MODEL_PATH --task generate \
    --trust-remote-code  --limit-mm-per-prompt image=99999 \
    --mm_processor_kwargs '{"max_pixels": 501760}' \
    --max-model-len 32768 --max-num-batched-tokens 65536 \
    --port 8201 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.9 \