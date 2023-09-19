#!/usr/bin/env bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

set -e

# Check if environment variable PRESIGNED_URL is set
if [[ -z "${PRESIGNED_URL}" ]]; then
    echo "PRESIGNED_URL environment variable is not set."
    exit 1
fi

# Get MODEL_STORE from command line argument
MODEL_STORE="$1"
if [[ -z "${MODEL_STORE}" ]]; then
    echo "Please provide the MODEL_STORE as the first command line argument."
    exit 1
fi

# Get MODEL_NAME from command line argument
MODEL_NAME="$2"
if [[ -z "${MODEL_NAME}" ]]; then
    echo "Please provide the MODEL_NAME as the second command line argument."
    exit 1
fi

# We'll use the MODEL_NAME to determine which model(s) to download
for m in ${MODEL_NAME//,/ }
do
    if [[ $m == "llama" ]]; then
        echo "Downloading common files for all models"

        # Ensure the llama directory exists
        mkdir -p ${MODEL_STORE}/${m}

        # Download LICENSE
        if [ ! -e ${MODEL_STORE}/${m}"/LICENSE" ]; then
            wget --continue ${PRESIGNED_URL/'*'/"LICENSE"} -O ${MODEL_STORE}/${m}"/LICENSE"
        fi

        # Download USE_POLICY.md
        if [ ! -e ${MODEL_STORE}/${m}"/USE_POLICY.md" ]; then
            wget --continue ${PRESIGNED_URL/'*'/"USE_POLICY.md"} -O ${MODEL_STORE}/${m}"/USE_POLICY.md"
        fi

        # Download tokenizer.model
        if [ ! -e ${MODEL_STORE}/${m}"/tokenizer.model" ]; then
            wget --continue ${PRESIGNED_URL/'*'/"tokenizer.model"} -O ${MODEL_STORE}/${m}"/tokenizer.model"
        fi

        # Download tokenizer_checklist.chk
        if [ ! -e ${MODEL_STORE}/${m}"/tokenizer_checklist.chk" ]; then
            wget --continue ${PRESIGNED_URL/'*'/"tokenizer_checklist.chk"} -O ${MODEL_STORE}/${m}"/tokenizer_checklist.chk"
        fi

        CPU_ARCH=$(uname -m)
        if [ "$CPU_ARCH" = "arm64" ]; then
            (cd ${MODEL_STORE}/${m} && md5 tokenizer_checklist.chk)
        else
            (cd ${MODEL_STORE}/${m} && md5sum -c tokenizer_checklist.chk)
        fi

        # Skip rest of the loop for "llama"
        continue
    fi

    if [[ $m == "llama-2-7b" ]]; then
        SHARD=0
        MODEL=$m
    elif [[ $m == "llama-2-7b-chat" ]]; then
        SHARD=0
        MODEL="llama-2-7b-chat"
    elif [[ $m == "llama-2-13b" ]]; then
        SHARD=1
        MODEL=$m
    elif [[ $m == "llama-2-13b-chat" ]]; then
        SHARD=1
        MODEL="llama-2-13b-chat"
    elif [[ $m == "llama-2-70b" ]]; then
        SHARD=7
        MODEL=$m
    elif [[ $m == "llama-2-70b-chat" ]]; then
        SHARD=7
        MODEL="llama-2-70b-chat"
    fi

    echo "Downloading ${MODEL}"
    mkdir -p ${MODEL_STORE}"/${MODEL}"

    for s in $(seq -f "0%g" 0 ${SHARD})
    do
        # Check if file exists before downloading
        if [ ! -e ${MODEL_STORE}"/${MODEL}/consolidated.${s}.pth" ]; then
            wget ${PRESIGNED_URL/'*'/"${MODEL}/consolidated.${s}.pth"} -O ${MODEL_STORE}"/${MODEL}/consolidated.${s}.pth"
        fi
    done

    # Check if file exists before downloading
    if [ ! -e ${MODEL_STORE}"/${MODEL}/params.json" ]; then
        wget --continue ${PRESIGNED_URL/'*'/"${MODEL}/params.json"} -O ${MODEL_STORE}"/${MODEL}/params.json"
    fi

    # Check if file exists before downloading
    if [ ! -e ${MODEL_STORE}"/${MODEL}/checklist.chk" ]; then
        wget --continue ${PRESIGNED_URL/'*'/"${MODEL}/checklist.chk"} -O ${MODEL_STORE}"/${MODEL}/checklist.chk"
    fi

    echo "Checking checksums"
    if [ "$CPU_ARCH" = "arm64" ]; then
      (cd ${MODEL_STORE}"/${MODEL}" && md5 checklist.chk)
    else
      (cd ${MODEL_STORE}"/${MODEL}" && md5sum -c checklist.chk)
    fi

done
