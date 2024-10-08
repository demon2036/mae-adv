export train_batch_size=4096 warmup_epoch=40 epoch=800

python3 src/main_pretrain_mae.py \
      --output-dir $GCS_MODEL_DIR/cifar10_mae_test/mae \
    --train-dataset-shards "$GCS_DATASET_DIR/imagenet-1k-wds/imagenet1k-train-{0000..1023}.tar" \
    --train-batch-size $train_batch_size \
    --train-loader-workers 80 \
    --random-crop rrc \
    --auto-augment none \
    --color-jitter 0.0 \
    --random-erasing 0.0 \
    --augment-repeats 1 \
    --test-crop-ratio 1.0 \
    --mixup 0.8 \
    --cutmix 1.0 \
    --criterion bce \
    --label-smoothing 0.0 \
    --layers 12 \
    --dim 192 \
    --heads 3 \
    --labels 1000 \
    --patch-size 16  \
    --image-size 224  \
    --posemb sincos2d \
    --pooling cls \
    --dropout 0.0 \
    --droppath 0.0 \
    --init-seed 0 \
    --mixup-seed 0 \
    --dropout-seed 0 \
    --shuffle-seed 0 \
    --optimizer adamw \
    --learning-rate 2.4e-3 \
    --weight-decay 0.05 \
    --adam-b1 0.9 \
    --adam-b2 0.95 \
    --adam-eps 1e-8 \
    --lr-decay 1.0 \
    --clip-grad 0.0 \
    --grad-accum 1 \
    --warmup-steps $((1281167 * $warmup_epoch / $train_batch_size)) \
    --training-steps $((1281167 * $epoch / $train_batch_size)) \
    --log-interval 100 \
    --eval-interval $((1281167 * 50 / $train_batch_size)) \
    --project deit3-jax-mae-cifar10 \
    --name $(basename $0 .sh) \
    --decoder_layers 1
#    --ipaddr $(curl -s ifconfig.me) \
#    --hostname $(hostname) \
