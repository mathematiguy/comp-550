# This Makefile automates routine tasks for this Singularity-based project.
REPO_NAME := $(shell basename `git rev-parse --show-toplevel` | tr '[:upper:]' '[:lower:]')
IMAGE := $(REPO_NAME).sif
RUN ?= singularity exec $(FLAGS) -B submodules/llama:/opt/llama $(IMAGE)
SINGULARITY_ARGS ?=
FLAGS ?=
VENV_PATH ?= venv

include cluster/makefile

.PHONY: test_llama convert_llama_weights clean jupyter container shell

test_llama:
	$(RUN) torchrun --nproc_per_node 1 submodules/llama/example_chat_completion.py \
		--ckpt_dir models/llama-2-7b-chat/ \
		--tokenizer_path models/llama/tokenizer.model \
		--max_seq_len 512 --max_batch_size 6

convert_llama_weights:
	$(RUN) python3 submodules/transformers/src/transformers/models/llama/convert_llama_weights_to_hf.py \
    --input_dir models/llama-2-7b --model_size 7B --output_dir hf_models/llama-2-7b

clean:
	rm -f report/*.blg report/*.fls report/*.out report/*.log report/*.fdb_latexmk report/*.aux report/*.pdf report/*.bbl report/*.toc

jupyter:
	$(RUN)  jupyter lab \
		--ip=0.0.0.0 \
		--no-browser \
		--port 8888

# Builds a Singularity container from the Singularity definition file.
# Note: This command requires sudo privileges.
container: $(IMAGE)
$(IMAGE): Singularity submodules/llama/README.md
	sudo singularity build $(IMAGE) $(SINGULARITY_ARGS) Singularity

submodules/llama/README.md:
	git submodule init && git submodule update

# Starts a shell within the Singularity container, with the virtual environment activated.
shell:
	singularity shell $(FLAGS) $(IMAGE) $(SINGULARITY_ARGS) bash
