#!/bin/bash

_serve() {
    local MODEL=$1
    local TENSOR_PARALLEL_SIZE=${2:-8}
    local MAX_MODEL_LEN=${3:-$(( 32 * 1024 ))}
    local IMAGE_LIMIT=${4:-1}
    local MAX_WAIT_TIME=3600
    shift 4

    local start_time
    start_time=$(date +%s)
    local tmp_log_file
    tmp_log_file=$(mktemp)

    vllm serve "$MODEL" \
        --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
        --data-parallel-size $(( 8 / "$TENSOR_PARALLEL_SIZE" )) \
        --gpu-memory-utilization 0.9 \
        --limit-mm-per-prompt "{\"image\":$IMAGE_LIMIT,\"video\":0}" \
        --max-model-len "$MAX_MODEL_LEN" \
        --mm-encoder-tp-mode data \
        --enforce-eager \
        --trust-remote-code \
        --mm-processor-cache-gb 64 \
        "$@" > "$tmp_log_file" 2>&1 &

    vllm_pid=$!
    sleep 1

    echo "vLLM for $MODEL is starting (PID: $vllm_pid)..."
    echo "Logging to $tmp_log_file"

    while true; do
        if ! kill -0 $vllm_pid >/dev/null 2>&1; then
            echo "ERROR: vllm serve failed: process died"
            exit 1
        fi

        local time_elapsed
        time_elapsed=$(($(date +%s) - start_time))
        if curl -s "http://localhost:8000/v1/models" | grep -q "$MODEL"; then
            echo "vLLM for $MODEL is ready (time elapsed: $time_elapsed seconds)"
            break
        fi

        if [[ $time_elapsed -gt $MAX_WAIT_TIME ]]; then
            echo "ERROR: vLLM for $MODEL is not ready after $MAX_WAIT_TIME seconds, trying to kill the process..."
            kill $vllm_pid
            wait $vllm_pid
            exit 1
        fi
        sleep 5
    done
}

_serve "$@"