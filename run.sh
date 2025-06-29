torchrun --nproc-per-node=1 run_beit3_finetuning.py  \
        --model beit3_base_patch16_480 \
        --input_size 480 \
        --task vqav2 \
        --batch_size 4 \
        --sentencepiece_model beit3.spm \
        --finetune beit3_base_indomain_patch16_224.pth \
        --data_path /home/pranav/v2_ExplanableAI/beit3/data \
        --output_dir /home/pranav/v2_ExplanableAI/beit3/abcd \
        --eval \
        --dist_eval \
        --checkpoint_activations \
        --eval_batch_size 4


